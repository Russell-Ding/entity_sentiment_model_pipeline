#!/usr/bin/env python3
"""Deterministic post-processing cleanup for the DeepSeek T1 labels.

Audit (Claude + Kimi cross-validation) flagged 3 systematic issues. This pass
handles the two that are fully deterministic; point 3 (person sentiment) is
done separately via a targeted Kimi re-label.

POINT 1 — Market/equity INDICES scored with non-zero sentiment.
  Indices (S&P 500, Dow, Nasdaq Composite, Nikkei, Hang Seng, Russell, VIX,
  FTSE, STOXX, DAX, CAC...) have no business stake — force sentiment_score -> 0.0.
  Trap avoided: index *providers* that are real companies keep their sentiment
  (MSCI Inc., S&P Global Inc., S&P Global Ratings) via an explicit name set +
  company-marker exclusion, never a loose "S&P"/"500" regex.

POINT 2 — Sidebar / "Most Read" boilerplate tail leakage.
  ~31% of articles append an unrelated headline block ("Most Read from
  Bloomberg", "Related Articles", "Recommended for you"...) stuffed with
  entities that aren't part of the story. We DELETE that tail:
    - truncate `text` at the marker,
    - drop any mention at/after the cut,
    - drop any entity left with no surviving mention.
  The marker must sit in the back `--tail-frac` of the text (default 0.5) so we
  never chop a real body that merely says "read more" early on.

Writes a NEW file (<input>.cleaned.jsonl); original untouched. Idempotent.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_IN = PROJECT / "data/labeled/deepseek_t1/labels_10ticker.jsonl"

# ---------------- POINT 1: indices ----------------
SENT_TYPES = {"ORG", "TICKER"}
INDEX_NAMES = {
    "s&p 500", "s&p500", "standard & poor's 500", "standard and poor's 500",
    "dow jones industrial average", "dow jones", "the dow", "djia", "dow",
    "nasdaq composite", "nasdaq 100", "nasdaq-100",
    "nikkei", "nikkei 225",
    "hang seng",
    "russell 2000", "russell 1000", "russell 3000",
    "ftse 100", "ftse",
    "euro stoxx 50", "stoxx 600", "stoxx europe 600",
    "dax", "cac 40", "cac",
    "vix", "cboe volatility", "cboe volatility i",
    "shanghai composite", "kospi", "sensex", "nifty 50", "tsx composite",
}
SP500_RE = re.compile(r"^s\s*&\s*p\s*500\b")
COMPANY_MARKERS = ("global", " inc", "inc.", "ratings", "corp", "llc", "ltd",
                   "holdings", "plc", " co.", " group", "msci")


def normalize(name: str) -> str:
    n = " ".join((name or "").lower().split())
    if n.endswith(" index"):
        n = n[: -len(" index")]
    return n.strip()


def is_index(ent: dict) -> bool:
    if ent.get("type") not in SENT_TYPES:
        return False
    raw = (ent.get("canonical_name") or "").lower()
    if any(mk in raw for mk in COMPANY_MARKERS):
        return False
    n = normalize(ent.get("canonical_name") or "")
    return n in INDEX_NAMES or bool(SP500_RE.match(n))


# ---------------- POINT 2: boilerplate tail ----------------
# High-precision tail markers only, ANCHORED to line start (^\s* in MULTILINE).
# Real boilerplate blocks always begin their own line ("\n\nMost Read from
# Bloomberg"); anchoring kills mid-body false positives like
# "coronavirus-related content". Ambiguous phrases ("more from", "read more")
# are excluded entirely so we never truncate a real article.
BOILER_RE = re.compile(
    r"^\s*(most read from|most read\b|related (articles|stories|quotes)|"
    r"recommended for you|you might (also )?like|trending now|popular (stories|now)|"
    r"read next|sign up for (our|the)|subscribe to (our|the))",
    re.I | re.M,
)
MENTION_KEYS = ("ner_mentions", "coref_mentions", "sentiment_expanded_mentions")


def find_cut(text: str, tail_frac: float) -> int | None:
    """Offset of the earliest high-precision marker in the back `tail_frac`."""
    if not text:
        return None
    floor = int(len(text) * (1.0 - tail_frac))
    best = None
    for m in BOILER_RE.finditer(text):
        if m.start() >= floor:
            best = m.start()
            break
    return best


def strip_tail(rec: dict, cut: int) -> tuple[int, int]:
    """Truncate text at cut, drop tail mentions, drop tail-only entities.
    Returns (entities_dropped, mentions_dropped)."""
    rec["text"] = rec.get("text", "")[:cut].rstrip()
    kept_ents, dropped_ents, dropped_ms = [], 0, 0
    for e in rec.get("entities", []):
        survived = False
        for k in MENTION_KEYS:
            ms = e.get(k)
            if not ms:
                continue
            keep = [m for m in ms if isinstance(m.get("end_char"), int) and m["end_char"] <= cut]
            dropped_ms += len(ms) - len(keep)
            e[k] = keep
            if keep:
                survived = True
        if survived:
            kept_ents.append(e)
        else:
            dropped_ents += 1
    rec["entities"] = kept_ents
    return dropped_ents, dropped_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--output", default=None, help="default: <input>.cleaned.jsonl")
    ap.add_argument("--tail-frac", type=float, default=0.5,
                    help="marker must be in the back this fraction of text (default 0.5)")
    args = ap.parse_args()
    src = Path(args.input)
    dst = Path(args.output) if args.output else src.with_suffix(".cleaned.jsonl")

    n_art = 0
    p1_zeroed = 0
    p1_names: dict[str, int] = {}
    p2_truncated = p2_ents_dropped = p2_ms_dropped = p2_chars = 0
    tail_examples: list[str] = []

    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            n_art += 1
            r = json.loads(line)

            # POINT 1
            for e in r.get("entities", []):
                if is_index(e) and e.get("sentiment_score") not in (None, 0, 0.0):
                    e["sentiment_score"] = 0.0
                    p1_zeroed += 1
                    k = e.get("canonical_name") or "?"
                    p1_names[k] = p1_names.get(k, 0) + 1

            # POINT 2
            text = r.get("text", "")
            cut = find_cut(text, args.tail_frac)
            if cut is not None:
                if len(tail_examples) < 6:
                    snippet = text[cut:cut + 70].replace("\n", " ")
                    tail_examples.append(f"  cut@{cut}/{len(text)}: …{text[max(0,cut-40):cut]}❚{snippet}…")
                de, dm = strip_tail(r, cut)
                p2_truncated += 1
                p2_ents_dropped += de
                p2_ms_dropped += dm
                p2_chars += len(text) - len(r["text"])

            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"articles processed : {n_art}\n")
    print(f"POINT 1 — index sentiments zeroed: {p1_zeroed}")
    for k, c in sorted(p1_names.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {c:5d}  {k}")
    print(f"\nPOINT 2 — boilerplate tails deleted: {p2_truncated} articles "
          f"({100*p2_truncated/n_art:.1f}%)")
    print(f"    entities dropped (tail-only): {p2_ents_dropped}")
    print(f"    mentions dropped            : {p2_ms_dropped}")
    print(f"    characters removed          : {p2_chars:,}")
    print("    cut examples (❚ = cut point):")
    for ex in tail_examples:
        print(ex)
    print(f"\nwrote -> {dst}")


if __name__ == "__main__":
    main()
