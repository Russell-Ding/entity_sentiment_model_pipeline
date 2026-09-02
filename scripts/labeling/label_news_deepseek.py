#!/usr/bin/env python3
"""Label Tier-1 news with DeepSeek V4 Pro to produce additional model training data.

DeepSeek twin of label_news_kimi.py. Calls the DeepSeek API (OpenAI-protocol
endpoint at api.deepseek.com, model deepseek-v4-pro) with the production
entity-labeling prompt (data_label_criteria/relabeling_prompt.md), using JSON
mode so output is always valid JSON. The model returns entities with TEXT-only
mentions; this script string-matches each mention back into the article to recover
char positions, producing records in the EXACT train.jsonl schema:

    {"id", "text", "entities": [{canonical_id, canonical_name, type, linked_ticker,
     linked_company, ner_mentions[{text,start_char,end_char}], coref_mentions[...],
     sentiment_expanded_mentions[], sentiment_score, is_sentiment_only}]}

so the output can be concatenated onto data/labeled/final/train.jsonl for a retrain.

Why DeepSeek: ~4.6x cheaper output than Kimi ($0.87 vs $4.00 / 1M) and pay-as-you-go
(no 5-hour quota wall), so it runs in hours instead of days.

Resumable + idempotent: re-running appends only not-yet-labeled ids, records
content-rejected ids to a sidecar, and survives crashes (partial-tail truncation).

Usage:
    # Pilot (validate format + cost on 20 articles)
    python3 scripts/labeling/label_news_deepseek.py --limit 20 --workers 8 \\
        --output data/labeled/deepseek_t1/pilot.jsonl

    # Full 10-ticker T1 run
    python3 scripts/labeling/label_news_deepseek.py --workers 16 \\
        --output data/labeled/deepseek_t1/labels_10ticker.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    import openai  # DeepSeek speaks the OpenAI protocol
except ImportError:
    print("ERROR: `pip install openai` (DeepSeek uses the OpenAI SDK).", file=sys.stderr)
    sys.exit(1)
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ds-label")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Use deepseek-chat: it is PINNED to v4-flash NON-THINKING mode (same v4-flash price,
# $0.14/$0.28 per 1M). The raw "deepseek-v4-flash" id intermittently auto-engages
# reasoning on dense articles — verified a 7.9k-char article burned all 16,384 output
# tokens on 58k chars of hidden reasoning_content and returned EMPTY content (unparseable),
# causing ~45% failures on a stretch of long articles. deepseek-chat labeled the identical
# article cleanly (finish=stop, 49 entities, no reasoning). (deepseek-chat is marked
# legacy/deprecated 2026-07-24, but this run completes well before then.)
DEFAULT_MODEL = "deepseek-chat"
PROMPT_PATH = PROJECT_ROOT / "data_label_criteria" / "relabeling_prompt.md"
NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "eodhd_bulk_20260518" / "news_retiered_v4"

DEFAULT_TICKERS = ["AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US", "NVDA.US",
                   "META.US", "TSLA.US", "BRK-B.US", "JPM.US", "V.US"]
SENTIMENT_TYPES = {"ORG", "TICKER", "PERSON"}
VALID_TYPES = {"ORG", "TICKER", "PERSON", "MONEY", "PERCENT", "DATE"}
MAX_TEXT_CHARS = 8000  # cap input (model only needs ~2048 tok of context); controls cost


# --------------------------------------------------------------------------- #
# API key + client
# --------------------------------------------------------------------------- #

def get_deepseek_key() -> Optional[str]:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    secrets = PROJECT_ROOT / "config" / "secrets.yaml"
    if secrets.exists() and HAS_YAML:
        try:
            data = yaml.safe_load(secrets.read_text())
            return (data or {}).get("api", {}).get("deepseek_v3_api")
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Article loading (T1, deduped, stubs excluded)
# --------------------------------------------------------------------------- #

def build_text(rec: Dict[str, Any]) -> str:
    """Match the inference-time text construction so labels align with model input."""
    title = (rec.get("title") or "").strip()
    content = (rec.get("content") or "").strip()
    text = (title + ". " + content).strip().strip(".").strip()
    return text


def load_t1_articles(tickers: List[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for tk in tickers:
        path = NEWS_DIR / f"{tk}.jsonl.gz"
        if not path.exists():
            logger.warning(f"missing news file: {path.name}")
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("final_source_tier") != 1:
                    continue
                if (rec.get("content_type") or "").strip().lower() == "mt_newswires":
                    continue  # paywall stub, no body to label
                aid = rec.get("id")
                if not aid or aid in seen:
                    continue
                text = build_text(rec)
                if len(text) < 30:
                    continue
                seen.add(aid)
                out.append({"id": aid, "text": text[:MAX_TEXT_CHARS]})
                if limit and len(out) >= limit:
                    return out
    return out


# --------------------------------------------------------------------------- #
# Response parsing + position recovery  (identical to the Kimi pipeline)
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_model_json(raw: str) -> Optional[Dict[str, Any]]:
    """JSON mode should return clean JSON, but keep a fence/outermost-object fallback."""
    s = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _is_valid_match(text: str, start: int, end: int, min_len_for_boundary: int = 3) -> bool:
    """For short substrings, require non-alphanumeric boundaries to avoid matching inside words."""
    if (end - start) > min_len_for_boundary:
        return True
    left_ok = start == 0 or not text[start - 1].isalnum()
    right_ok = end == len(text) or not text[end].isalnum()
    return left_ok and right_ok


def find_all_positions(text: str, substring: str) -> List[Tuple[int, int]]:
    if not substring:
        return []
    out, start = [], 0
    while True:
        pos = text.find(substring, start)
        if pos == -1:
            break
        end = pos + len(substring)
        if _is_valid_match(text, pos, end):
            out.append((pos, end))
        start = pos + 1
    return out


def to_training_entity(ent: Dict[str, Any], text: str, article_id: str) -> Optional[Dict[str, Any]]:
    """Convert one model entity (text-only mentions) to the train.jsonl entity schema."""
    etype = (ent.get("type") or "").strip().upper()
    if etype not in VALID_TYPES:
        logger.warning(f"[{article_id}] dropping entity with invalid type: {etype}")
        return None
    ner_mentions = []
    for mt in ent.get("ner_mention_texts", []) or []:
        if not isinstance(mt, str):
            continue
        mt_clean = mt.strip()
        for s, e in find_all_positions(text, mt_clean):
            ner_mentions.append({"text": mt_clean, "start_char": s, "end_char": e})
    coref_mentions = []
    for mt in ent.get("coref_mention_texts", []) or []:
        if not isinstance(mt, str):
            continue
        mt_clean = mt.strip()
        pos = find_all_positions(text, mt_clean)
        if pos:
            s, e = pos[0]
            coref_mentions.append({"text": mt_clean, "start_char": s, "end_char": e})
    if not ner_mentions:
        cname = ent.get("canonical_name") or ent.get("canonical_id") or "?"
        logger.warning(f"[{article_id}] dropping entity '{cname}' ({etype}): no NER mentions matched in text")
        return None
    sent = ent.get("sentiment_score")
    if etype not in SENTIMENT_TYPES:
        sent = None
    elif sent is not None:
        try:
            sent = float(sent)
            sent = None if math.isnan(sent) else max(-1.0, min(1.0, sent))
        except (TypeError, ValueError):
            sent = None
    canonical_id = str(ent.get("canonical_id") or "").strip()
    canonical_name = str(ent.get("canonical_name") or "").strip()
    if not canonical_id:
        canonical_id = canonical_name
    if not canonical_id:
        canonical_id = f"unknown_{hash(article_id + str(ent.get('type'))) & 0xFFFFFFFF:08x}"
    if not canonical_name:
        canonical_name = canonical_id
    return {
        "canonical_id": canonical_id,
        "canonical_name": canonical_name,
        "type": etype,
        "linked_ticker": ent.get("linked_ticker"),
        "linked_company": ent.get("linked_company"),
        "ner_mentions": ner_mentions,
        "coref_mentions": coref_mentions,
        "sentiment_expanded_mentions": [],
        "sentiment_score": sent,
        "is_sentiment_only": False,
    }


class DeepSeekLabeler:
    def __init__(self, api_key: str, system_prompt: str, model: str = DEFAULT_MODEL,
                 max_tokens: int = 4096, max_retries: int = 3, request_timeout: float = 120.0):
        self.client = openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=request_timeout)
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self.in_tokens = 0
        self.out_tokens = 0
        self.cache_hit_tokens = 0

    def label(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        base_user = f'Label this article. Output ONE JSON object only.\n\nid: {article["id"]}\ntext:\n{article["text"]}'
        for attempt in range(1, self.max_retries + 1):
            user = base_user
            if attempt > 1:
                user = base_user + "\n\nCRITICAL: Output ONLY a single valid JSON object. No markdown, no prose."
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},  # DeepSeek JSON mode -> always valid JSON
                )
            except openai.BadRequestError as e:
                # 400 = deterministic (content refusal, or malformed/oversized). Mark the
                # article permanently REJECTED so it is recorded and skipped, not retried.
                logger.warning(f"[{article['id']}] REJECTED (400, will skip permanently): {str(e)[:120]}")
                return {"_rejected": True, "id": article["id"], "reason": str(e)[:300]}
            except openai.RateLimitError:
                # 429 = rate limited; fail fast so the breaker trips and the supervisor waits.
                # Article is NOT rejected -> returns None and is retried later.
                return None
            except Exception as e:  # network / 5xx — transient, retry with backoff
                if attempt == self.max_retries:
                    logger.warning(f"[{article['id']}] API failed after {attempt} tries: {e}")
                    return None
                time.sleep(min((2 ** attempt) + random.random(), 60.0))
                continue
            usage = getattr(resp, "usage", None)
            if usage is not None:
                with self._lock:
                    self.in_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.out_tokens += getattr(usage, "completion_tokens", 0) or 0
                    self.cache_hit_tokens += getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            raw = (resp.choices[0].message.content or "") if resp.choices else ""
            parsed = parse_model_json(raw)
            if parsed is None:
                if attempt == self.max_retries:
                    logger.warning(f"[{article['id']}] unparseable JSON after {attempt} tries")
                    return None
                continue
            ents = []
            for e in parsed.get("entities", []) or []:
                te = to_training_entity(e, article["text"], article["id"])
                if te:
                    ents.append(te)
            return {"id": article["id"], "text": article["text"], "entities": ents}
        return None


# --------------------------------------------------------------------------- #
# Driver  (identical resume / rejection / watchdog logic to the Kimi pipeline)
# --------------------------------------------------------------------------- #

def scan_and_truncate(path: Path) -> set:
    """Return set of labeled ids and truncate the file to the last valid newline."""
    done: set[str] = set()
    if not path.exists():
        return done
    valid_offset = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                rec = json.loads(line.decode("utf-8"))
                done.add(rec["id"])
                valid_offset = f.tell()
            except Exception:
                break
    if valid_offset < path.stat().st_size:
        logger.warning(f"Truncating {path.name} from {path.stat().st_size:,} to {valid_offset:,} bytes (partial last line)")
        with path.open("r+b") as f:
            f.truncate(valid_offset)
    return done


def write_status(path: Path, status: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".status.tmp")
    try:
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        tmp.rename(path)
    except Exception:
        pass


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser(description="Label T1 news with DeepSeek for training data")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--limit", type=int, default=None, help="cap articles (pilot mode)")
    ap.add_argument("--workers", type=int, default=16, help="concurrent API calls")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="deepseek-v4-pro | deepseek-v4-flash")
    # Generous cap so dense articles (~8k output) don't truncate; you only pay for tokens
    # actually generated, so a high ceiling just prevents the ~10% truncation failures.
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--force", action="store_true", help="ignore existing output, relabel all")
    ap.add_argument("--prompt", type=Path, default=PROMPT_PATH,
                    help=f"labeling prompt file (default: {PROMPT_PATH.name})")
    ap.add_argument("--status-file", type=Path, default=None)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    ap.add_argument("--stall-timeout", type=float, default=900.0,
                    help="abort if no successful label for this many seconds")
    ap.add_argument("--max-consecutive-failures", type=int, default=40)
    args = ap.parse_args()
    status_path = args.status_file or args.output.with_suffix(".status.json")

    api_key = get_deepseek_key()
    if not api_key:
        logger.error("No deepseek key (env DEEPSEEK_API_KEY or config/secrets.yaml api.deepseek_v3_api)")
        sys.exit(1)
    if not args.prompt.exists():
        logger.error(f"Prompt not found: {args.prompt}")
        sys.exit(1)
    system_prompt = args.prompt.read_text(encoding="utf-8")
    logger.info(f"Prompt: {args.prompt.name}")

    logger.info(f"Model: {args.model}")
    logger.info(f"Loading T1 articles from {len(args.tickers)} tickers...")
    articles = load_t1_articles(args.tickers, args.limit)
    logger.info(f"Loaded {len(articles):,} unique T1 articles")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = args.output.with_suffix(".rejected.jsonl")
    done = set() if args.force else scan_and_truncate(args.output)
    rejected_ids = set()
    if not args.force and rejected_path.exists():
        for line in rejected_path.open(encoding="utf-8"):
            if line.strip():
                try:
                    rejected_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    skip = done | rejected_ids
    if skip:
        before = len(articles)
        articles = [a for a in articles if a["id"] not in skip]
        logger.info(f"Resuming: {len(done):,} labeled, {len(rejected_ids):,} rejected (skipped), "
                    f"{len(articles):,} remaining (was {before:,})")
    if not articles:
        logger.info("Nothing to label.")
        return

    labeler = DeepSeekLabeler(api_key, system_prompt, model=args.model,
                              max_tokens=args.max_tokens, request_timeout=args.request_timeout)
    n_ok = n_fail = n_empty = n_rejected = 0
    consecutive_failures = 0
    last_success = time.time()
    abort_reason = None
    t0 = time.time()
    remaining = len(articles)

    def emit_status(processed: int, finished: bool = False) -> None:
        elapsed = time.time() - t0
        rate = processed / max(elapsed, 1e-9)
        with labeler._lock:
            itok, otok, chit = labeler.in_tokens, labeler.out_tokens, labeler.cache_hit_tokens
        write_status(status_path, {
            "updated": _now_iso(),
            "model": args.model,
            "processed_this_run": processed,
            "remaining_this_run": remaining - processed,
            "labeled_total_on_disk": len(done) + n_ok,
            "ok": n_ok, "failed": n_fail, "empty_entities": n_empty,
            "rejected_high_risk": n_rejected,
            "consecutive_failures": consecutive_failures,
            "rate_per_sec": round(rate, 3),
            "eta_min": round((remaining - processed) / max(rate, 1e-9) / 60, 1),
            "elapsed_min": round(elapsed / 60, 1),
            "tokens_in": itok, "tokens_out": otok, "tokens_cache_hit": chit,
            "finished": finished,
            "abort_reason": abort_reason,
        })

    file_mode = "w" if args.force else "a"
    with open(args.output, file_mode, encoding="utf-8", buffering=1 << 20) as fout, \
         open(rejected_path, file_mode, encoding="utf-8") as frej:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            article_iter = iter(articles)
            pending: dict = {}
            total_submitted = 0

            def refill() -> None:
                nonlocal total_submitted
                while len(pending) < args.workers * 4 and total_submitted < len(articles):
                    try:
                        a = next(article_iter)
                    except StopIteration:
                        break
                    pending[ex.submit(labeler.label, a)] = a
                    total_submitted += 1

            refill()
            processed = 0
            while pending and abort_reason is None:
                done_futs, _ = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    a = pending.pop(fut)
                    rec = fut.result()
                    if rec is None:
                        n_fail += 1
                        consecutive_failures += 1
                    elif rec.get("_rejected"):
                        frej.write(json.dumps({"id": rec["id"], "reason": rec["reason"]}, ensure_ascii=False) + "\n")
                        frej.flush()
                        n_rejected += 1
                        consecutive_failures = 0
                        last_success = time.time()
                    else:
                        if not rec["entities"]:
                            n_empty += 1
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_ok += 1
                        consecutive_failures = 0
                        last_success = time.time()
                    processed += 1

                if consecutive_failures >= args.max_consecutive_failures:
                    abort_reason = f"{consecutive_failures} consecutive failures (rate-limit/API down)"
                elif time.time() - last_success > args.stall_timeout:
                    abort_reason = f"no successful label for {args.stall_timeout:.0f}s (stall)"

                if processed % 50 == 0:
                    fout.flush()
                    emit_status(processed)
                    rate = processed / max(time.time() - t0, 1e-9)
                    with labeler._lock:
                        itok, otok = labeler.in_tokens, labeler.out_tokens
                    logger.info(f"  {processed:,}/{remaining:,}  ok={n_ok} fail={n_fail} rejected={n_rejected} empty={n_empty}  "
                                f"{rate:.2f} art/s  ETA {(remaining-processed)/max(rate,1e-9)/60:.0f}m  "
                                f"tok in/out={itok:,}/{otok:,}")
                refill()
            fout.flush()

    dt = time.time() - t0
    finished_clean = abort_reason is None
    emit_status(processed, finished=finished_clean)
    if abort_reason:
        logger.error(f"ABORTED after {dt/60:.1f}m: {abort_reason}")
        logger.error(f"  Labeled {n_ok} this run. Re-run the SAME command to resume from here.")
    else:
        logger.info(f"Done in {dt/60:.1f}m. ok={n_ok} fail={n_fail} rejected={n_rejected} empty_entities={n_empty}")
    logger.info(f"Tokens: input={labeler.in_tokens:,} (cache-hit {labeler.cache_hit_tokens:,}) output={labeler.out_tokens:,}")
    logger.info(f"Output: {args.output}  Status: {status_path}")
    sys.exit(0 if finished_clean else 2)


if __name__ == "__main__":
    main()
