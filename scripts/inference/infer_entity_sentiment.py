#!/usr/bin/env python3
"""Run v2.0 entity-sentiment inference on retiered news articles.

Reads a single ticker's <TICKER>.jsonl.gz from data/raw/<bulk>/news_retiered_v4/,
filters by source tier, and runs the production v2.0 model end-to-end:
text → NER (CRF Viterbi) → sentiment head → per-entity aggregation.

Output is one JSONL record per article with embedded entity list:
    {
      "article_id": ..., "date": ..., "trade_date": "YYYY-MM-DD",
      "primary_ticker": ..., "source_tier": 1, "content_type": "reuters",
      "detected_source": ..., "url": ...,
      "n_spans_total": 12, "n_entities": 4,
      "entities": [
        {"canonical_id": "Apple", "entity_type": "COMPANY",
         "mentions": [{"text": "Apple", "char_start": 0, "char_end": 5}, ...],
         "num_mentions": 5, "sentiment": 0.62, "sentiment_std": 0.04}, ...]
    }

Usage:
    # Sanity test on first 50 T1 AAPL articles (local CPU/MPS)
    python3 scripts/inference/infer_entity_sentiment.py \\
        --input data/raw/eodhd_bulk_20260518/news_retiered_v4/AAPL.US.jsonl.gz \\
        --output outputs/inference/AAPL.US.t1_sentiment_sanity.jsonl \\
        --max-articles 50

    # Full T1 for one ticker (Colab GPU recommended)
    python3 scripts/inference/infer_entity_sentiment.py \\
        --input data/raw/eodhd_bulk_20260518/news_retiered_v4/AAPL.US.jsonl.gz \\
        --output outputs/inference/AAPL.US.t1_sentiment.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "evaluation"))

# Reuse the production-validated 2-pass NER + sentiment batched predictor from
# the evaluation script. That predictor is the same code path used to compute
# our reported e2e Pearson r = 0.6279 / NER F1 = 0.4586 holdout metrics.
from evaluate_e2e_pipeline import load_model, predict_articles_batched, predict_article  # noqa: E402
from training.preprocessing import SENTIMENT_ENTITY_TYPES  # noqa: E402
from transformers import LongformerTokenizerFast  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
for h in logging.getLogger().handlers:
    if hasattr(h, "stream"):
        h.stream = sys.stdout
logging.getLogger().handlers[0].flush = lambda: sys.stdout.flush()
logger = logging.getLogger("infer")

DEFAULT_CHECKPOINT = PROJECT_ROOT / "trained_model" / "v2.0_20260517" / "model.pt"
MAX_LENGTH = 2048
CHUNK_SIZE = 64  # articles between progress logs / chunked outer loop

# NER noise filter — drops obvious artifacts (BIO-decode tails, single-char
# spans, generic suffix words). Tightened type-by-type below.
_TICKER_RE = re.compile(r"^\$?[A-Z]{1,5}(?:\.[A-Z]{1,3})?$")
_GENERIC_COMPANY_WORDS = {
    "group", "inc", "corp", "co", "ltd", "company", "corporation",
    "holdings", "holding", "limited", "plc", "llc",
}


def is_valid_span(text: str, entity_type: str) -> bool:
    """Reject NER artifacts before sentiment aggregation.

    Rules:
      - All types: must contain >=2 alphabetic characters.
      - TICKER: must match a ticker-shaped pattern (1-5 caps, optional .XX suffix).
      - COMPANY/ORG/PERSON: must contain at least one uppercase letter, and the
        first non-space character must be uppercase (catches lead-char-dropped
        artifacts like "ow Moutai Co.,").
      - COMPANY/ORG: reject standalone generic suffix words like "Group" or
        "Inc" that are not real entity names.
    """
    t = (text or "").strip()
    if not t:
        return False
    if sum(1 for c in t if c.isalpha()) < 2:
        return False
    if entity_type == "TICKER":
        return bool(_TICKER_RE.match(t))
    if not any(c.isupper() for c in t):
        return False
    if not t[0].isupper():
        return False
    if entity_type in {"COMPANY", "ORG"} and t.lower() in _GENERIC_COMPANY_WORDS:
        return False
    return True


def resolve_device(arg: str) -> str:
    """Resolve 'auto' to cuda > mps > cpu, else honor the user's choice."""
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_articles(
    input_path: Path,
    tier: int = 1,
    max_articles: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load articles from a retiered <ticker>.jsonl.gz, filtered by tier."""
    articles: List[Dict[str, Any]] = []
    n_total = 0
    n_kept = 0
    n_bad = 0
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                n_bad += 1
                if n_bad <= 5:
                    logger.warning(f"Skipping malformed JSONL line {n_total}: {e}")
                continue
            if rec.get("final_source_tier") != tier:
                continue
            title = (rec.get("title") or "").strip()
            content = (rec.get("content") or "").strip()
            # The training data uses "title. content" concatenation; match that here
            # so the encoder sees the same input shape as during training.
            text = (title + ". " + content).strip().strip(".").strip()
            if not text:
                continue
            articles.append({
                "id": rec["id"],
                "text": text,
                "date": rec.get("date"),
                "trade_date": (rec.get("date") or "")[:10] or None,
                "url": rec.get("url"),
                "primary_ticker": rec.get("primary_ticker"),
                "source_tier": rec.get("final_source_tier"),
                "content_type": rec.get("content_type"),
                "rule_name": rec.get("rule_name"),
                "detected_source": rec.get("detected_source"),
                "symbols": rec.get("symbols", []),
            })
            n_kept += 1
            if max_articles and n_kept >= max_articles:
                break
    if n_bad:
        logger.warning(f"Skipped {n_bad} malformed JSONL line(s)")
    logger.info(f"Filtered {n_kept}/{n_total} articles (tier T{tier}) from {input_path.name}")
    return articles


def aggregate_spans_to_entities(
    spans: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter NER noise, then group surviving spans by surface form and
    mean-pool sentiment per group.

    Surface-form grouping is a stand-in for true canonical resolution. Two
    surface forms of the same entity ("Apple" and "Apple Inc.") will appear as
    separate canonical_ids here; downstream aggregation can do alias resolution.

    Returns (entities, n_filtered) — the filtered count is span-level garbage
    rejected by `is_valid_span`, not the spans dropped for not being a
    sentiment-bearing type.
    """
    n_filtered = 0
    groups: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {
        "type": None,
        "mentions": [],
        "sentiments": [],
    })
    for s in spans:
        etype = s.get("type")
        if etype not in SENTIMENT_ENTITY_TYPES:
            continue
        raw = (s.get("text") or "").strip()
        if not is_valid_span(raw, etype):
            n_filtered += 1
            continue
        key = (raw, etype)
        groups[key]["type"] = etype
        groups[key]["mentions"].append({
            "text": raw,
            "char_start": s["char_start"],
            "char_end": s["char_end"],
        })
        if s.get("pred_sentiment") is not None:
            groups[key]["sentiments"].append(float(s["pred_sentiment"]))

    out: List[Dict[str, Any]] = []
    for key, g in groups.items():
        sentiments = g["sentiments"]
        n_sents = len(sentiments)
        if n_sents == 0:
            mean_sent: Optional[float] = None
            sent_std: Optional[float] = None
        else:
            mean_sent = sum(sentiments) / n_sents
            if n_sents > 1:
                sent_std = (sum((x - mean_sent) ** 2 for x in sentiments) / n_sents) ** 0.5
            else:
                sent_std = 0.0
        out.append({
            "canonical_id": key[0],
            "entity_type": g["type"],
            "mentions": g["mentions"],
            "num_mentions": len(g["mentions"]),
            "sentiment": round(mean_sent, 4) if mean_sent is not None else None,
            "sentiment_std": round(sent_std, 4) if sent_std is not None else None,
        })
    # Sort by mention count (most-referenced entity first)
    out.sort(key=lambda e: -e["num_mentions"])
    return out, n_filtered


def _load_done_ids(path: Path) -> Set[str]:
    """Read existing JSONL output and return the set of already-processed article IDs."""
    done: Set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = rec.get("article_id")
            if aid:
                done.add(str(aid))
    return done


def _write_progress(path: Path, done_ids: Set[str], total: int) -> None:
    """Atomic progress write: temp file then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done_ids": sorted(done_ids), "n_done": len(done_ids), "n_total": total}, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2.0 entity-sentiment inference on retiered news")
    parser.add_argument("--input", type=Path, required=True,
                        help="<ticker>.jsonl.gz from news_retiered_v4/")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSONL output path (one record per article)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)})")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="final_source_tier to filter by (default: 1)")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Cap articles for sanity testing (default: no cap)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Articles per encoder forward (default: 4)")
    parser.add_argument("--local-output-dir", type=Path, default=None,
                        help="Local dir to write outputs first (Colab pattern). "
                             "If unset, writes directly to --output.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output instead of resuming.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    logger.info(f"Device: {device}")

    # Determine local vs Drive paths
    if args.local_output_dir:
        local_output_dir = args.local_output_dir
        local_output_dir.mkdir(parents=True, exist_ok=True)
        local_output_path = local_output_dir / args.output.name
        progress_path = local_output_dir / (args.output.name + ".progress.json")
        # Add file logger
        fh = logging.FileHandler(str(local_output_dir / (args.output.name + ".log")))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        logging.getLogger().addHandler(fh)
    else:
        local_output_dir = None
        local_output_path = args.output
        progress_path = args.output.parent / (args.output.name + ".progress.json")

    # Load articles first (fail fast on missing input)
    articles = load_articles(args.input, tier=args.tier, max_articles=args.max_articles)
    if not articles:
        logger.error("No articles match the tier filter. Exiting.")
        sys.exit(1)

    # Resume logic
    done_ids: Set[str] = set()
    if not args.force and local_output_path.exists():
        done_ids = _load_done_ids(local_output_path)
        logger.info(f"Resuming: found {len(done_ids)} already-processed articles in {local_output_path}")

    if done_ids:
        articles = [a for a in articles if str(a["id"]) not in done_ids]
        logger.info(f"Remaining articles to process: {len(articles)}")
        if not articles:
            logger.info("All articles already processed. Nothing to do.")
            # Still attempt Drive sync if local != output
            if local_output_dir and local_output_path.exists():
                args.output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(str(local_output_path), str(args.output))
                    logger.info(f"Synced existing local output to Drive: {args.output}")
                except Exception as e:
                    logger.warning(f"Drive sync failed: {e}")
            sys.exit(0)

    # Load model
    model, epoch, val_metrics = load_model(args.checkpoint, device)
    logger.info(f"Model loaded (epoch={epoch})")
    if val_metrics:
        for k in ("ner_f1", "sentiment_corr", "sentiment_pearson", "pearson_r"):
            if k in val_metrics:
                logger.info(f"  val.{k} = {val_metrics[k]}")

    tokenizer = LongformerTokenizerFast.from_pretrained(model.encoder_name)

    # Open output: append if resuming, else write
    mode = "a" if done_ids else "w"
    n_processed = len(done_ids)
    n_with_entity = 0
    n_with_sentiment = 0
    n_total_entities = 0
    n_total_spans = 0
    n_total_filtered = 0

    t_start = time.time()
    with open(local_output_path, mode, encoding="utf-8") as fout:
        for chunk_start in range(0, len(articles), CHUNK_SIZE):
            chunk = articles[chunk_start:chunk_start + CHUNK_SIZE]

            # Batched inference with fallback
            try:
                results = predict_articles_batched(
                    model, tokenizer, chunk, device,
                    max_length=MAX_LENGTH, mini_batch_size=args.batch_size,
                )
            except (RuntimeError, IndexError, KeyError, ValueError, TypeError) as e:
                logger.warning(
                    f"Batch starting at {chunk_start}: {type(e).__name__}: {e}. "
                    f"Falling back to per-article inference for this chunk."
                )
                results = []
                for art in chunk:
                    try:
                        spans, _diag = predict_article(
                            model, tokenizer, art.get("text", ""), device,
                            max_length=MAX_LENGTH, ner_mode="single-pass",
                        )
                        results.append((spans, _diag))
                    except Exception as e2:
                        logger.warning(f"Article {art['id']}: per-article fallback failed: {e2}")
                        results.append(([], {}))

            for art, (spans, _diag) in zip(chunk, results):
                entities, n_filtered = aggregate_spans_to_entities(spans)
                n_total_spans += len(spans)
                n_total_entities += len(entities)
                n_total_filtered += n_filtered
                if entities:
                    n_with_entity += 1
                if entities and any(e["sentiment"] is not None for e in entities):
                    n_with_sentiment += 1

                record = {
                    "article_id": art["id"],
                    "date": art["date"],
                    "trade_date": art["trade_date"],
                    "primary_ticker": art["primary_ticker"],
                    "source_tier": art["source_tier"],
                    "content_type": art["content_type"],
                    "detected_source": art.get("detected_source"),
                    "url": art["url"],
                    "symbols": art.get("symbols", []),
                    "n_spans_total": len(spans),
                    "n_entities": len(entities),
                    "entities": entities,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                done_ids.add(str(art["id"]))
                n_processed += 1

            done = chunk_start + len(chunk)
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Progress: {done}/{len(articles)} ({done/len(articles)*100:5.1f}%)  "
                f"rate={rate:.2f} art/s"
            )

            # Persist progress sidecar atomically
            if local_output_dir:
                _write_progress(progress_path, done_ids, len(articles) + len(done_ids))

    logger.info(f"Done. Wrote {n_processed} records to {local_output_path}")
    if n_processed:
        logger.info(f"  Articles with >=1 entity:           {n_with_entity:6d} ({n_with_entity/n_processed*100:5.1f}%)")
        logger.info(f"  Articles with >=1 scored sentiment: {n_with_sentiment:6d} ({n_with_sentiment/n_processed*100:5.1f}%)")
        logger.info(f"  Total spans detected:               {n_total_spans:6d}  (avg {n_total_spans/n_processed:.1f}/article)")
        logger.info(f"  NER-noise spans filtered:           {n_total_filtered:6d}  ({n_total_filtered/max(n_total_spans,1)*100:5.1f}% of detected spans)")
        logger.info(f"  Total unique entities:              {n_total_entities:6d}  (avg {n_total_entities/n_processed:.1f}/article)")

    # Sync to Drive (local-first pattern)
    if local_output_dir:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(local_output_path), str(args.output))
            logger.info(f"Synced output to Drive: {args.output}")
        except Exception as e:
            logger.warning(f"Drive sync failed: {e}")
            logger.info(f"Local output is safe at: {local_output_path}")


if __name__ == "__main__":
    main()
