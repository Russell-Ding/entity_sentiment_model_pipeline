#!/usr/bin/env python3
"""Re-score PERSON sentiment with Kimi (audit point 3).

The DeepSeek labels are training-grade for ORG/TICKER but the cross-validation
audit flagged PERSON sentiment for sign errors / nulls. This pass keeps the
extracted persons and ONLY re-scores them, via Kimi (the stronger auditor on the
nuanced person-fate calls).

Cheaper than a full relabel: we send the article text + the already-extracted
person list and ask for one score per person (tiny output). We do NOT re-extract.

Resumable side-file design — never mutates the cleaned dataset:
    input : data/labeled/deepseek_t1/labels_10ticker.cleaned.jsonl
    output: data/labeled/deepseek_t1/person_scores.jsonl   (one line per article)
            {"id": "...", "scores": {"<canonical_name>": 0.4, ...}}
A separate merge step (apply_person_scores.py) overlays these onto the cleaned
file to produce the final training set.

Usage:
    # pilot — measure speed / quality / refusal rate / tokens on 30 articles
    python3 scripts/labeling/relabel_persons_kimi.py --limit 30 --workers 4
    # full run
    python3 scripts/labeling/relabel_persons_kimi.py --workers 8
"""
from __future__ import annotations
import argparse, json, logging, random, sys, threading, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT = Path(__file__).resolve().parents[2]
try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic", file=sys.stderr); sys.exit(1)
try:
    import yaml
except ImportError:
    yaml = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("kimi-person")

KIMI_MODEL = "kimi-for-coding"
KIMI_BASE_URL = "https://api.kimi.com/coding"
IN_DEFAULT = PROJECT / "data/labeled/deepseek_t1/labels_10ticker.cleaned.jsonl"
OUT_DEFAULT = PROJECT / "data/labeled/deepseek_t1/person_scores.jsonl"
PROMPT_PATH = PROJECT / "data_label_criteria/person_sentiment_prompt.md"
MAX_TEXT_CHARS = 9000


def get_kimi_key() -> Optional[str]:
    import os
    if os.environ.get("KIMI_API_KEY"):
        return os.environ["KIMI_API_KEY"]
    sec = PROJECT / "config" / "secrets.yaml"
    if yaml and sec.exists():
        data = yaml.safe_load(sec.read_text()) or {}
        return (data.get("api", {}) or {}).get("kimi_api_key")
    return None


def parse_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    a, b = raw.find("{"), raw.rfind("}")
    if a == -1 or b == -1:
        return None
    try:
        return json.loads(raw[a:b + 1])
    except json.JSONDecodeError:
        return None


def person_list(rec: dict) -> List[dict]:
    """Distinct PERSON entities (by canonical_name), with a context cue."""
    seen, out = set(), []
    for e in rec.get("entities", []):
        if e.get("type") != "PERSON":
            continue
        name = e.get("canonical_name") or ""
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(e)
    return out


class PersonScorer:
    def __init__(self, api_key: str, system_prompt: str, max_tokens: int = 1024,
                 max_retries: int = 3, request_timeout: float = 120.0):
        self.client = anthropic.Anthropic(api_key=api_key, base_url=KIMI_BASE_URL, timeout=request_timeout)
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self.in_tokens = self.out_tokens = 0

    def score(self, rec: dict, persons: List[dict]) -> Optional[dict]:
        names = [p.get("canonical_name") for p in persons]
        listing = "\n".join(f"{i}: {n}" for i, n in enumerate(names))
        text = rec.get("text", "")[:MAX_TEXT_CHARS]
        base = (f"Article text:\n{text}\n\nPeople to score (echo each index i):\n{listing}\n\n"
                f"Output ONLY the JSON object with a score for every index.")
        for attempt in range(1, self.max_retries + 1):
            user = base if attempt == 1 else base + "\n\nCRITICAL: output ONLY raw JSON, no prose."
            try:
                resp = self.client.messages.create(
                    model=KIMI_MODEL, max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": user}],
                )
            except anthropic.BadRequestError as e:
                return {"_rejected": True, "id": rec["id"], "reason": str(e)[:200]}
            except anthropic.RateLimitError:
                return None
            except Exception as e:
                if attempt == self.max_retries:
                    log.warning(f"[{rec['id']}] API failed: {e}"); return None
                time.sleep(min(2 ** attempt + random.random(), 60.0)); continue
            with self._lock:
                self.in_tokens += getattr(resp.usage, "input_tokens", 0) or 0
                self.out_tokens += getattr(resp.usage, "output_tokens", 0) or 0
            parsed = parse_json("".join(getattr(b, "text", "") for b in resp.content))
            if parsed is None or "scores" not in parsed:
                if attempt == self.max_retries:
                    log.warning(f"[{rec['id']}] unparseable"); return None
                continue
            scores: Dict[str, float] = {}
            for item in parsed.get("scores", []):
                try:
                    i = int(item["i"]); s = float(item["s"])
                except (KeyError, ValueError, TypeError):
                    continue
                if 0 <= i < len(names):
                    scores[names[i]] = max(-1.0, min(1.0, round(s, 2)))
            return {"id": rec["id"], "scores": scores}
        return None


def scan_done(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    valid = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                done.add(json.loads(line.decode())["id"]); valid = f.tell()
            except Exception:
                break
    if valid < path.stat().st_size:
        with path.open("r+b") as f:
            f.truncate(valid)
    return done


def write_status(path: Path, st: dict) -> None:
    tmp = path.with_suffix(".status.tmp")
    try:
        tmp.write_text(json.dumps(st, indent=2)); tmp.rename(path)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    ap.add_argument("--stall-timeout", type=float, default=900.0)
    ap.add_argument("--max-consecutive-failures", type=int, default=40)
    args = ap.parse_args()

    key = get_kimi_key()
    if not key:
        log.error("No kimi_api_key (env KIMI_API_KEY or config/secrets.yaml api.kimi_api_key)"); sys.exit(1)
    system_prompt = args.prompt.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    status_path = args.output.with_suffix(".status.json")
    rejected_path = args.output.with_suffix(".rejected.jsonl")

    done = scan_done(args.output)
    log.info(f"Resume: {len(done)} articles already scored")

    # Content-policy "high risk" 400s are DETERMINISTIC — they re-fail identically
    # every resume. Skip already-rejected ids so we don't re-burn quota on them
    # each quota-wall cycle. (Their persons simply keep the DeepSeek score.)
    rejected_ids: set = set()
    if rejected_path.exists():
        for line in rejected_path.open(encoding="utf-8"):
            try:
                rejected_ids.add(json.loads(line)["id"])
            except Exception:
                continue
    if rejected_ids:
        done |= rejected_ids
        log.info(f"Resume: skipping {len(rejected_ids)} previously-rejected (high-risk) ids")

    # Build worklist: only articles that have >=1 PERSON and aren't done.
    work: List[tuple] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["id"] in done:
                continue
            persons = person_list(rec)
            if persons:
                work.append((rec, persons))
            if args.limit and len(work) >= args.limit:
                break
    log.info(f"To score: {len(work)} articles")
    if not work:
        log.info("Nothing to do."); return

    scorer = PersonScorer(key, system_prompt, max_tokens=args.max_tokens,
                          request_timeout=args.request_timeout)
    out_f = args.output.open("a", encoding="utf-8")
    rej_f = rejected_path.open("a", encoding="utf-8")
    t0 = time.time()
    last_success = time.time()
    n_ok = n_fail = n_rej = consec = 0
    n_persons = 0
    lock = threading.Lock()

    def handle(res):
        nonlocal n_ok, n_fail, n_rej, consec, last_success, n_persons
        if res is None:
            n_fail += 1; consec += 1; return
        if res.get("_rejected"):
            n_rej += 1; consec = 0; last_success = time.time()
            rej_f.write(json.dumps(res) + "\n"); rej_f.flush(); return
        n_ok += 1; consec = 0; last_success = time.time()
        n_persons += len(res.get("scores", {}))
        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
        if n_ok % 50 == 0:
            out_f.flush()
            rate = n_ok / max(time.time() - t0, 1e-9)
            remaining = (len(work) - n_ok - n_rej) / max(rate, 1e-9) / 60
            write_status(status_path, {
                "updated": time.strftime("%F %T"), "scored": n_ok, "rejected": n_rej,
                "failed": n_fail, "consecutive_failures": consec,
                "persons_scored": n_persons, "rate_per_sec": round(rate, 3),
                "eta_min": round(remaining, 1), "in_tokens": scorer.in_tokens,
                "out_tokens": scorer.out_tokens, "to_score": len(work),
            })
            log.info(f"ok={n_ok} rej={n_rej} fail={n_fail} {rate:.2f}/s eta={remaining:.0f}m persons={n_persons}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        it = iter(work)
        for _ in range(min(args.workers * 2, len(work))):
            rec, persons = next(it)
            futures[ex.submit(scorer.score, rec, persons)] = True
        while futures:
            dn, _ = wait(list(futures), timeout=30, return_when=FIRST_COMPLETED)
            for fu in dn:
                del futures[fu]
                try:
                    with lock:
                        handle(fu.result())
                except Exception as e:
                    log.warning(f"worker error: {e}"); n_fail += 1; consec += 1
                try:
                    rec, persons = next(it)
                    futures[ex.submit(scorer.score, rec, persons)] = True
                except StopIteration:
                    pass
            if consec >= args.max_consecutive_failures:
                log.error(f"ABORT: {consec} consecutive failures (quota/API). Resume later."); break
            if time.time() - last_success > args.stall_timeout:
                log.error(f"ABORT: no success in {args.stall_timeout}s (stall)."); break

    out_f.flush(); out_f.close(); rej_f.close()
    elapsed = time.time() - t0
    write_status(status_path, {
        "updated": time.strftime("%F %T"), "scored": n_ok, "rejected": n_rej,
        "failed": n_fail, "persons_scored": n_persons,
        "in_tokens": scorer.in_tokens, "out_tokens": scorer.out_tokens,
        "finished": True, "elapsed_min": round(elapsed / 60, 1),
    })
    aborted = consec >= args.max_consecutive_failures or (time.time() - last_success > args.stall_timeout)
    log.info(f"DONE ok={n_ok} rej={n_rej} fail={n_fail} persons={n_persons} "
             f"in={scorer.in_tokens:,} out={scorer.out_tokens:,} {elapsed/60:.1f}m")
    sys.exit(2 if aborted else 0)


if __name__ == "__main__":
    main()
