#!/usr/bin/env python3
"""One-shot quality check on a random sample of DeepSeek-labeled articles.

Computes OBJECTIVE deterioration metrics (no LLM — fast, deterministic, reliable as
an auto-stop). If any metric breaches its threshold, optionally kills the labeling
run and writes a WARNING file. Prints a single status line:

    QUALITY_OK    <metrics...>
    QUALITY_BREACH <reason> | <metrics...>

Designed to be called on a loop (e.g. every 10 min) by a Monitor. The metrics catch
the real degradation modes: output-format break, regression to all-neutral sentiment,
under-extraction, empty records, and failure/refusal spikes.

Baselines (from the v2-decisive prompt pilot): ~20 entities/article, ~66% neutral on
sentiment-bearing entities, ~100% position correctness, near-0 empty records.
"""
from __future__ import annotations
import argparse, json, os, random, signal, statistics, subprocess, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data/labeled/deepseek_t1/labels_10ticker.jsonl"
STATUS = PROJECT / "data/labeled/deepseek_t1/labels_10ticker.status.json"
WARNING = PROJECT / "data/labeled/deepseek_t1/QUALITY_WARNING.txt"
SENT_TYPES = {"ORG", "TICKER", "PERSON"}

# Thresholds — a breach on ANY trips the alarm.
TH = {
    "min_positions_ok": 0.95,    # text[s:e]==mention; < this = format/parse broke
    "max_neutral_rate": 0.88,    # > this = regressing to all-neutral (baseline ~0.66)
    "min_entities_per_article": 8.0,   # < this = under-extraction (baseline ~20)
    "max_empty_rate": 0.25,      # > this = many records with 0 entities
    "max_fail_rate": 0.15,       # from status: failed/processed
    "max_reject_rate": 0.20,     # content refusals spiking
}


def reservoir_sample_lines(path: Path, k: int) -> list:
    """Random k records from a JSONL without loading it all (reservoir over lines)."""
    sample = []
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if len(sample) < k:
                sample.append(line)
            else:
                j = random.randint(0, n - 1)
                if j < k:
                    sample[j] = line
    recs = []
    for line in sample:
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return recs, n


def check(sample_size: int) -> tuple[bool, str, dict]:
    if not OUT.exists() or OUT.stat().st_size == 0:
        return True, "no output yet", {}
    recs, total = reservoir_sample_lines(OUT, sample_size)
    if not recs:
        return True, "no parseable records yet", {"total": total}

    pos_ok = pos_tot = 0
    sent_neutral = sent_tot = 0
    ent_counts = []
    empty = 0
    for r in recs:
        ents = r.get("entities", [])
        ent_counts.append(len(ents))
        if not ents:
            empty += 1
        text = r.get("text", "")
        for e in ents:
            for m in e.get("ner_mentions", []) + e.get("coref_mentions", []):
                pos_tot += 1
                s, en = m.get("start_char"), m.get("end_char")
                if isinstance(s, int) and isinstance(en, int) and text[s:en] == m.get("text"):
                    pos_ok += 1
            if e.get("type") in SENT_TYPES and e.get("sentiment_score") is not None:
                sent_tot += 1
                if abs(e["sentiment_score"]) < 0.05:
                    sent_neutral += 1

    m = {
        "sampled": len(recs), "total_labeled": total,
        "positions_ok": round(pos_ok / pos_tot, 3) if pos_tot else 1.0,
        "neutral_rate": round(sent_neutral / sent_tot, 3) if sent_tot else 0.0,
        "entities_per_article": round(statistics.mean(ent_counts), 1) if ent_counts else 0.0,
        "empty_rate": round(empty / len(recs), 3),
    }
    # failure/reject rates from the status sidecar
    if STATUS.exists():
        try:
            s = json.loads(STATUS.read_text())
            proc = s.get("processed_this_run", 0)
            # Only evaluate fail/reject RATES once enough has been processed this run.
            # Right after a resume, processed_this_run is tiny, so 2-3 transient JSON
            # failures spike the ratio and would false-trip the breaker.
            if proc >= 50:
                m["fail_rate"] = round(s.get("failed", 0) / proc, 3)
                m["reject_rate"] = round(s.get("rejected_high_risk", 0) / proc, 3)
            else:
                m["proc_this_run"] = proc  # informational only; rates not enforced yet
        except Exception:
            pass

    reasons = []
    if m["positions_ok"] < TH["min_positions_ok"]:
        reasons.append(f"positions_ok={m['positions_ok']}<{TH['min_positions_ok']}")
    if m["neutral_rate"] > TH["max_neutral_rate"]:
        reasons.append(f"neutral_rate={m['neutral_rate']}>{TH['max_neutral_rate']}")
    if m["entities_per_article"] < TH["min_entities_per_article"]:
        reasons.append(f"entities/article={m['entities_per_article']}<{TH['min_entities_per_article']}")
    if m["empty_rate"] > TH["max_empty_rate"]:
        reasons.append(f"empty_rate={m['empty_rate']}>{TH['max_empty_rate']}")
    if m.get("fail_rate", 0) > TH["max_fail_rate"]:
        reasons.append(f"fail_rate={m['fail_rate']}>{TH['max_fail_rate']}")
    if m.get("reject_rate", 0) > TH["max_reject_rate"]:
        reasons.append(f"reject_rate={m['reject_rate']}>{TH['max_reject_rate']}")

    return (len(reasons) == 0), "; ".join(reasons), m


def kill_run() -> None:
    for pat in ("run_deepseek_labeling.sh", "label_news_deepseek.py"):
        try:
            subprocess.run(["pkill", "-f", pat], timeout=10)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=10)
    ap.add_argument("--enforce", action="store_true", help="kill the run + write WARNING on breach")
    ap.add_argument("--min-labeled", type=int, default=100, help="skip checks until this many labeled (warmup)")
    args = ap.parse_args()

    ok, reason, m = check(args.sample_size)
    mstr = " ".join(f"{k}={v}" for k, v in m.items())

    # During warmup (too few labeled), don't enforce — just report.
    if m.get("total_labeled", 0) < args.min_labeled:
        print(f"QUALITY_OK (warmup) {mstr}", flush=True)
        sys.exit(0)

    if ok:
        print(f"QUALITY_OK {mstr}", flush=True)
        sys.exit(0)

    # BREACH
    if args.enforce:
        kill_run()
        WARNING.write_text(
            f"QUALITY BREACH at {time.strftime('%F %T')}\nReasons: {reason}\nMetrics: {mstr}\n"
            f"Run was STOPPED. Investigate before resuming.\n", encoding="utf-8")
    print(f"QUALITY_BREACH {reason} | {mstr}", flush=True)
    sys.exit(2)


if __name__ == "__main__":
    main()
