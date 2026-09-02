#!/usr/bin/env python3
"""Label Tier-1 news with Kimi K2 to produce additional model training data.

Calls the Kimi API (Anthropic-protocol endpoint at api.kimi.com/coding, model
kimi-for-coding) with the production entity-labeling prompt
(data_label_criteria/relabeling_prompt.md). Kimi returns entities with TEXT-only
mentions; this script string-matches each mention back into the article to recover
char positions, producing records in the EXACT train.jsonl schema:

    {"id", "text", "entities": [{canonical_id, canonical_name, type, linked_ticker,
     linked_company, ner_mentions[{text,start_char,end_char}], coref_mentions[...],
     sentiment_expanded_mentions[], sentiment_score, is_sentiment_only}]}

so the output can be concatenated onto data/labeled/final/train.jsonl for a retrain.

Scope: Tier-1 articles for a set of tickers (default: the 10 already inferred).
Articles are deduplicated by id across feeds. mt_newswires paywall stubs are skipped.

Resumable: re-running appends only not-yet-labeled article ids. Concurrent API
calls via a thread pool. Output is flushed in batches (every 50 records).

Usage:
    # Pilot (validate format + cost on 5 articles)
    python3 scripts/labeling/label_news_kimi.py --limit 5 --workers 2 \\
        --output data/labeled/kimi_t1/pilot.jsonl

    # Full 10-ticker T1 run
    python3 scripts/labeling/label_news_kimi.py --workers 8 \\
        --output data/labeled/kimi_t1/labels_10ticker.jsonl
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
    import anthropic
except ImportError:
    print("ERROR: `pip install anthropic` (the SDK speaks the Kimi endpoint too).", file=sys.stderr)
    sys.exit(1)
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("kimi-label")

KIMI_MODEL = "kimi-for-coding"
KIMI_BASE_URL = "https://api.kimi.com/coding"
PROMPT_PATH = PROJECT_ROOT / "data_label_criteria" / "relabeling_prompt.md"
NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "eodhd_bulk_20260518" / "news_retiered_v4"

DEFAULT_TICKERS = ["AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US", "NVDA.US",
                   "META.US", "TSLA.US", "BRK-B.US", "JPM.US", "V.US"]
SENTIMENT_TYPES = {"ORG", "TICKER", "PERSON"}
VALID_TYPES = {"ORG", "TICKER", "PERSON", "MONEY", "PERCENT", "DATE"}
MAX_TEXT_CHARS = 8000  # cap input (model context is ~2048 tok ≈ 8k chars); controls cost


# --------------------------------------------------------------------------- #
# API key + client
# --------------------------------------------------------------------------- #

def get_kimi_key() -> Optional[str]:
    if os.environ.get("KIMI_API_KEY"):
        return os.environ["KIMI_API_KEY"]
    secrets = PROJECT_ROOT / "config" / "secrets.yaml"
    if secrets.exists() and HAS_YAML:
        try:
            data = yaml.safe_load(secrets.read_text())
            return (data or {}).get("api", {}).get("kimi_api_key")
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
# Kimi call + response parsing
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_kimi_json(raw: str) -> Optional[Dict[str, Any]]:
    """Kimi may wrap output in markdown fences or add prose. Extract the JSON object."""
    s = _FENCE_RE.sub("", raw).strip()
    # Fast path
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the outermost {...}
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
    """Convert one Kimi entity (text-only mentions) to the train.jsonl entity schema."""
    etype = (ent.get("type") or "").strip().upper()
    if etype not in VALID_TYPES:
        logger.warning(f"[{article_id}] dropping entity with invalid type: {etype}")
        return None
    # NER mentions: find every occurrence of each mention text
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
        return None  # cannot place this entity in the text — drop
    sent = ent.get("sentiment_score")
    if etype not in SENTIMENT_TYPES:
        sent = None
    elif sent is not None:
        try:
            sent = float(sent)
            if math.isnan(sent):
                sent = None
            else:
                sent = max(-1.0, min(1.0, sent))
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


class KimiLabeler:
    def __init__(self, api_key: str, system_prompt: str, max_tokens: int = 4096,
                 max_retries: int = 3, request_timeout: float = 120.0):
        # request_timeout: a hung call fails after this many seconds instead of
        # blocking a worker forever (critical for the unattended multi-hour run).
        self.client = anthropic.Anthropic(api_key=api_key, base_url=KIMI_BASE_URL, timeout=request_timeout)
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self.in_tokens = 0
        self.out_tokens = 0

    def label(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        base_user = f'Label this article. Output ONE JSON line only.\n\nid: {article["id"]}\ntext:\n{article["text"]}'
        for attempt in range(1, self.max_retries + 1):
            user = base_user
            if attempt > 1:
                user = base_user + "\n\nCRITICAL: Your previous response was not valid JSON. Output ONLY raw JSON. No markdown fences, no explanation."
            try:
                resp = self.client.messages.create(
                    model=KIMI_MODEL,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": user}],
                )
            except anthropic.BadRequestError as e:
                # 400 = deterministic (content-safety "high risk", or malformed/oversized).
                # Retrying never helps and re-fails every resume cycle, so mark the article
                # permanently REJECTED so it is recorded and skipped, not retried.
                logger.warning(f"[{article['id']}] REJECTED (400, will skip permanently): {str(e)[:120]}")
                return {"_rejected": True, "id": article["id"], "reason": str(e)[:300]}
            except anthropic.RateLimitError:
                # 429 = quota exhausted for the period — deterministic until refresh.
                # Fail FAST (no in-process retry) so the consecutive-failure breaker trips
                # quickly and the supervisor gets to its wait-for-quota-refresh cycle. The
                # article is NOT rejected — it returns None and is retried after the refresh.
                return None
            except Exception as e:  # network / 5xx — transient, retry
                if attempt == self.max_retries:
                    logger.warning(f"[{article['id']}] API failed after {attempt} tries: {e}")
                    return None
                sleep_time = min((2 ** attempt) + random.random(), 60.0)
                time.sleep(sleep_time)
                continue
            with self._lock:
                self.in_tokens += getattr(resp.usage, "input_tokens", 0) or 0
                self.out_tokens += getattr(resp.usage, "output_tokens", 0) or 0
            raw = "".join(getattr(b, "text", "") for b in resp.content)
            parsed = parse_kimi_json(raw)
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
# Driver
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
    """Atomically write a tiny status JSON (so a monitor reads bytes, not the huge log)."""
    tmp = path.with_suffix(".status.tmp")
    try:
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        tmp.rename(path)
    except Exception:
        pass  # status is best-effort; never let it crash the run


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser(description="Label T1 news with Kimi for training data")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--limit", type=int, default=None, help="cap articles (pilot mode)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--force", action="store_true", help="ignore existing output, relabel all")
    ap.add_argument("--status-file", type=Path, default=None,
                    help="compact status JSON (default: <output>.status.json)")
    ap.add_argument("--request-timeout", type=float, default=120.0,
                    help="per-API-call timeout in seconds")
    ap.add_argument("--stall-timeout", type=float, default=900.0,
                    help="abort if no successful label for this many seconds (quota/API death)")
    ap.add_argument("--max-consecutive-failures", type=int, default=40,
                    help="abort after this many consecutive failures (quota exhausted)")
    args = ap.parse_args()
    status_path = args.status_file or args.output.with_suffix(".status.json")

    api_key = get_kimi_key()
    if not api_key:
        logger.error("No kimi_api_key (env KIMI_API_KEY or config/secrets.yaml api.kimi_api_key)")
        sys.exit(1)
    if not PROMPT_PATH.exists():
        logger.error(f"Prompt not found: {PROMPT_PATH}")
        sys.exit(1)
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    logger.info(f"Loading T1 articles from {len(args.tickers)} tickers...")
    articles = load_t1_articles(args.tickers, args.limit)
    logger.info(f"Loaded {len(articles):,} unique T1 articles")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = args.output.with_suffix(".rejected.jsonl")
    done = set() if args.force else scan_and_truncate(args.output)
    # Permanently-rejected ids (content-policy 400s) are skipped, never retried.
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

    labeler = KimiLabeler(api_key, system_prompt, max_tokens=args.max_tokens,
                          request_timeout=args.request_timeout)
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
            itok, otok = labeler.in_tokens, labeler.out_tokens
        write_status(status_path, {
            "updated": _now_iso(),
            "processed_this_run": processed,
            "remaining_this_run": remaining - processed,
            "labeled_total_on_disk": len(done) + n_ok,
            "ok": n_ok, "failed": n_fail, "empty_entities": n_empty,
            "rejected_high_risk": n_rejected,
            "consecutive_failures": consecutive_failures,
            "rate_per_sec": round(rate, 3),
            "eta_min": round((remaining - processed) / max(rate, 1e-9) / 60, 1),
            "elapsed_min": round(elapsed / 60, 1),
            "tokens_in": itok, "tokens_out": otok,
            "finished": finished,
            "abort_reason": abort_reason,
        })

    # --force starts fresh ("w" truncates); otherwise append to the resumed file.
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
                # timeout so the stall watchdog fires even if every call hangs
                done_futs, _ = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    a = pending.pop(fut)
                    rec = fut.result()
                    if rec is None:
                        n_fail += 1
                        consecutive_failures += 1
                    elif rec.get("_rejected"):
                        # content-policy 400: record so it is never retried; not a stall
                        frej.write(json.dumps({"id": rec["id"], "reason": rec["reason"]}, ensure_ascii=False) + "\n")
                        frej.flush()
                        n_rejected += 1
                        consecutive_failures = 0
                        last_success = time.time()  # a rejection is handled progress, not a stall
                    else:
                        if not rec["entities"]:
                            n_empty += 1
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_ok += 1
                        consecutive_failures = 0
                        last_success = time.time()
                    processed += 1

                # --- watchdogs (quota wall / dead API) ---
                if consecutive_failures >= args.max_consecutive_failures:
                    abort_reason = f"{consecutive_failures} consecutive failures (quota/API down)"
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
    logger.info(f"Tokens: input={labeler.in_tokens:,} output={labeler.out_tokens:,}")
    logger.info(f"Output: {args.output}  Status: {status_path}")
    sys.exit(0 if finished_clean else 2)  # exit 2 signals the supervisor to wait + resume


if __name__ == "__main__":
    main()
