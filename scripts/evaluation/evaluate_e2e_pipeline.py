#!/usr/bin/env python3
"""End-to-end pipeline evaluation: raw text -> NER -> sentiment -> match against gold.

Unlike evaluate_holdout_stage3.py (which uses gold entity positions), this runs
the full inference pipeline:
    1. Tokenize raw text only
    2. NER head predicts entity spans (CRF Viterbi decode)
    3. Build entity_masks from predicted spans
    4. Sentiment head produces per-span sentiment
    5. Match predicted spans to gold mentions (char-level IoU)
    6. Aggregate predicted sentiment per gold canonical entity (mean-pool)
    7. Report span-level NER F1, coverage, and sentiment metrics on covered entities

Saves results to outputs/e2e_evaluation/ with timestamped filenames. Designed to
run on Colab with the local-first save pattern (writes to /content/, then syncs
to Drive). Auto-terminates the runtime when complete.

Usage:
    python scripts/evaluation/evaluate_e2e_pipeline.py
    python scripts/evaluation/evaluate_e2e_pipeline.py --max-samples 200 --device cpu
    python scripts/evaluation/evaluate_e2e_pipeline.py \\
        --checkpoint checkpoints/stage3_sentiment_large/best_model.pt \\
        --benchmark data/labeled/final/holdout_relabeled.jsonl
"""

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

# transformers torch.load safety workaround (older transformers complains about
# weights_only=False; we know our checkpoints are trusted).
try:
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline import FinancialEntitySentimentModel
from models.sentiment_head import SentimentHead
from training.preprocessing import (
    LABEL_TO_ID, ID_TO_LABEL, SENTIMENT_ENTITY_TYPES,
)
from transformers import LongformerTokenizerFast


ENCODER_NAME = "allenai/longformer-large-4096"
HIDDEN_SIZE = 1024
MAX_LENGTH = 2048
NUM_NER_LABELS = 15

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model_20260418.pt"
DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "labeled" / "final" / "holdout_relabeled.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e2e_evaluation"

SENTIMENT_BUCKETS = [
    ("very_negative", -1.0, -0.6),
    ("negative",      -0.6, -0.2),
    ("neutral",       -0.2,  0.2),
    ("positive",       0.2,  0.6),
    ("very_positive",  0.6,  1.01),
]
BUCKET_NAMES = [b[0] for b in SENTIMENT_BUCKETS]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
for h in logging.getLogger().handlers:
    if hasattr(h, "stream"):
        h.stream = sys.stdout
logging.getLogger().handlers[0].flush = lambda: sys.stdout.flush()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path, device):
    """Load a SentimentHead (cross-attention) checkpoint.

    Old V1 self-attention checkpoints (Apr 2026 era) are no longer supported;
    they are archived under checkpoints/archive/ for reference only.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Sanity check: this script only supports the cross-attention sentiment head
    if any(k.startswith("sentiment_head.entity_attention.") for k in state_dict):
        raise RuntimeError(
            "Checkpoint contains the legacy V1 self-attention sentiment head "
            "(sentiment_head.entity_attention.*). This eval script only supports "
            "the V2 cross-attention head used in trained_model/v2.0_20260517/. "
            "Use git history (pre-2026-05-17) if you need to load V1 checkpoints."
        )

    model = FinancialEntitySentimentModel(
        encoder_name=ENCODER_NAME,
        hidden_size=HIDDEN_SIZE,
        num_ner_labels=NUM_NER_LABELS,
        use_ner_head=True,
        use_coref_head=False,
        use_crf_ner=True,
        ner_label_to_id=LABEL_TO_ID,
        device="cpu",
        max_length=MAX_LENGTH,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_metrics = checkpoint.get("val_metrics", {})

    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, epoch, val_metrics


# ---------------------------------------------------------------------------
# NER decoding: BIO predictions -> character-level spans
# ---------------------------------------------------------------------------
def decode_bio_spans(predictions, attention_mask, offset_mapping):
    """Convert CRF predictions to entity spans with both token and character positions.

    Args:
        predictions: 1D tensor of predicted label IDs (seq_len,)
        attention_mask: 1D tensor of attention mask
        offset_mapping: list of (char_start, char_end) per token

    Returns:
        list of dicts {type, token_start, token_end, char_start, char_end, text_range}
    """
    spans = []
    current = None
    valid_len = int(attention_mask.sum().item())

    for j in range(valid_len):
        pred_id = int(predictions[j].item())
        label = ID_TO_LABEL.get(pred_id, "O")
        offset = offset_mapping[j]
        # Special tokens (CLS/SEP/pad) have offset (0,0)
        is_special = (offset[0] == 0 and offset[1] == 0)

        if label == "O" or is_special:
            if current is not None:
                spans.append(current)
                current = None
            continue

        if label.startswith("B-"):
            if current is not None:
                spans.append(current)
            current = {
                "type": label[2:],
                "token_start": j,
                "token_end": j + 1,
                "char_start": offset[0],
                "char_end": offset[1],
            }
        elif label.startswith("I-"):
            entity_type = label[2:]
            if current is not None and current["type"] == entity_type:
                current["token_end"] = j + 1
                current["char_end"] = offset[1]
            else:
                # Invalid I- (no preceding B- of same type). End any current span,
                # treat this token as starting a new entity.
                if current is not None:
                    spans.append(current)
                current = {
                    "type": entity_type,
                    "token_start": j,
                    "token_end": j + 1,
                    "char_start": offset[0],
                    "char_end": offset[1],
                }

    if current is not None:
        spans.append(current)

    return spans


def build_entity_masks_for_spans(spans, seq_len, device):
    """Build [n_entities, seq_len] mask tensor from a list of token spans."""
    n = len(spans)
    if n == 0:
        return None
    masks = torch.zeros(n, seq_len, device=device)
    for i, span in enumerate(spans):
        masks[i, span["token_start"]:span["token_end"]] = 1.0
    return masks


# ---------------------------------------------------------------------------
# End-to-end forward pass for a single article
# ---------------------------------------------------------------------------
def _run_ner_with_global(model, input_ids, attention_mask, global_attn):
    """Run encoder + NER head with the given global attention mask.

    Returns:
        predictions: (seq_len,) Viterbi-decoded label IDs (or argmax for non-CRF)
        logits:      (seq_len, num_labels) raw logits for confidence estimation
    """
    encoder_output = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attn,
    )
    if model.use_crf_ner:
        ner_output = model.ner_head(encoder_output, attention_mask=attention_mask)
        predictions = ner_output["predictions"][0]
        logits = ner_output["logits"][0]
    else:
        ner_logits = model.ner_head(encoder_output)
        logits = ner_logits[0]
        predictions = logits.argmax(dim=-1)
    return predictions, logits


def _confident_global_mask(pass1_spans, pass1_logits, ner_confidence, seq_len, device):
    """Build a global-attention mask from Pass 1 entity tokens above a confidence threshold.

    For each Pass 1 span, all of its tokens must have softmax max-prob >=
    threshold for the span to seed Pass 2 global attention. This prevents
    low-confidence hallucinations from corrupting Pass 2 representations.

    Returns:
        high_conf_global: (seq_len,) {0,1} mask
        n_confident_tokens: int
    """
    probs = pass1_logits.softmax(dim=-1)
    max_probs = probs.max(dim=-1).values  # (seq_len,)
    high_conf = torch.zeros(seq_len, dtype=torch.long, device=device)
    for span in pass1_spans:
        ts, te = span["token_start"], span["token_end"]
        if (max_probs[ts:te] >= ner_confidence).all():
            high_conf[ts:te] = 1
    return high_conf, int(high_conf.sum().item())


@torch.no_grad()
def predict_articles_batched(model, tokenizer, articles, device,
                             max_length=MAX_LENGTH, mini_batch_size=8):
    """Batched single-pass inference for the e2e pipeline.

    Pass 1 (NER, CLS-only global) and Pass 2 (sentiment, entity-aware global)
    are each computed once per mini-batch of articles. The encoder runs at
    full GPU utilization instead of one article at a time.

    Only used in single-pass NER mode (the dropout-trained model handles
    CLS-only natively, so 2-pass refinement isn't needed).

    Args:
        articles: list of dicts with "text" and "id" keys.
        mini_batch_size: number of articles per encoder forward.

    Returns:
        list of (pred_spans, diag) tuples, one per input article. Preserves
        input order. Articles with empty text return ([], {...}).
    """
    n_total = len(articles)
    out: list = [None] * n_total  # type: ignore

    # Pre-fill empty/missing-text articles
    work_indices = []
    for i, art in enumerate(articles):
        text = art.get("text", "")
        if not text.strip():
            out[i] = ([], {"pass1_n_spans": 0, "pass1_n_high_conf_tokens": 0,
                          "used_pass2_ner": False, "pass1_zero_high_conf": True,
                          "empty_text": True})
        else:
            work_indices.append(i)

    # Mini-batches over the non-empty articles
    for batch_start in range(0, len(work_indices), mini_batch_size):
        batch_idxs = work_indices[batch_start:batch_start + mini_batch_size]
        batch_texts = [articles[i].get("text", "") for i in batch_idxs]
        bs = len(batch_idxs)

        encoding = tokenizer(
            batch_texts,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)           # (B, T)
        attention_mask = encoding["attention_mask"].to(device) # (B, T)
        offset_mappings = encoding["offset_mapping"]           # (B, T, 2)
        seq_len = input_ids.shape[1]

        # ----- Pass 1: NER (CLS-only global) -----
        global_attn_1 = torch.zeros_like(input_ids)
        global_attn_1[:, 0] = 1

        encoder_output_1 = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attn_1,
        )
        if model.use_crf_ner:
            ner_output = model.ner_head(encoder_output_1, attention_mask=attention_mask)
            pass1_preds = ner_output["predictions"]  # (B, T)
        else:
            pass1_preds = model.ner_head(encoder_output_1).argmax(dim=-1)  # (B, T)

        # Decode each article's spans on CPU
        batch_spans = []
        for b in range(bs):
            spans = decode_bio_spans(
                pass1_preds[b], attention_mask[b], offset_mappings[b].tolist()
            )
            batch_spans.append(spans)

        # Free encoder_output_1 before Pass 2 to save memory
        del encoder_output_1, pass1_preds

        # ----- Build sentiment inputs across the batch -----
        # Collect (batch_position, span_index, token_mask) for every sentiment-bearing span
        sent_batch_pos = []          # which sample in the mini-batch
        sent_span_idx = []           # which span within that sample
        sent_entity_masks = []       # (T,) per entity

        # Per-article global attention mask for Pass 2 (CLS + predicted entity tokens)
        global_attn_2 = torch.zeros_like(input_ids)
        global_attn_2[:, 0] = 1

        for b, spans in enumerate(batch_spans):
            for j, s in enumerate(spans):
                if s["type"] in SENTIMENT_ENTITY_TYPES:
                    mask = torch.zeros(seq_len, dtype=torch.float, device=device)
                    mask[s["token_start"]:s["token_end"]] = 1.0
                    sent_batch_pos.append(b)
                    sent_span_idx.append(j)
                    sent_entity_masks.append(mask)
                    global_attn_2[b, s["token_start"]:s["token_end"]] = 1

        # ----- Pass 2: sentiment encoder forward + head -----
        if sent_entity_masks:
            encoder_output_2 = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attn_2,
            )  # (B, T, H)

            # Gather per-entity context (each entity sees its own article's hidden states)
            sent_pos_t = torch.tensor(sent_batch_pos, device=device, dtype=torch.long)
            entity_context = encoder_output_2[sent_pos_t]    # (sum_K, T, H)
            entity_mask_t = torch.stack(sent_entity_masks)   # (sum_K, T)

            scores = model.sentiment_head(entity_context, entity_mask_t)  # (sum_K,)
            scores_cpu = scores.cpu().tolist()

            del encoder_output_2, entity_context, entity_mask_t

            # Distribute scores back to spans
            for k, score in enumerate(scores_cpu):
                b = sent_batch_pos[k]
                j = sent_span_idx[k]
                batch_spans[b][j]["pred_sentiment"] = float(score)

        # ----- Attach text and default null sentiment to remaining spans -----
        for b in range(bs):
            text = batch_texts[b]
            for s in batch_spans[b]:
                s["text"] = text[s["char_start"]:s["char_end"]]
                if "pred_sentiment" not in s:
                    s["pred_sentiment"] = None

        # Per-article diagnostics
        for local_b, art_idx in enumerate(batch_idxs):
            spans = batch_spans[local_b]
            diag = {
                "pass1_n_spans": len(spans),
                "pass1_n_high_conf_tokens": 0,  # not computed in batched single-pass
                "used_pass2_ner": False,        # this is single-pass only
                "pass1_zero_high_conf": (len(spans) == 0),
            }
            out[art_idx] = (spans, diag)

    return out


@torch.no_grad()
def predict_article(model, tokenizer, text, device, max_length=MAX_LENGTH,
                    ner_mode="two-pass", ner_confidence=0.7):
    """Run the full inference pipeline on one article.

    NER mode:
      - "single-pass": Pass 1 NER (CLS-only global). Train/test mismatch.
      - "two-pass":    Pass 1 NER (CLS-only) -> filter by confidence ->
                       Pass 2 NER (CLS + confident entity tokens as global) ->
                       use Pass 2 predictions for sentiment.

    Sentiment is always run on a separate encoder pass with global attention
    on all final-prediction entity tokens (training regime).

    MPS fallback: if a RuntimeError mentions 'mps', the article is rerun on CPU.

    Returns:
        pred_spans: list of {type, token_start, token_end, char_start, char_end,
                             text, pred_sentiment} dicts
        diag: per-article diagnostics dict
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].tolist()
    seq_len = input_ids.shape[1]

    diag = {
        "pass1_n_spans": 0,
        "pass1_n_high_conf_tokens": 0,
        "used_pass2_ner": False,
        "pass1_zero_high_conf": False,
    }

    def _ner_with_mps_fallback(global_attn):
        try:
            return _run_ner_with_global(model, input_ids, attention_mask, global_attn)
        except RuntimeError as e:
            if "mps" in str(e).lower() and device.type != "cpu":
                model.cpu()
                preds, logs = _run_ner_with_global(
                    model, input_ids.cpu(), attention_mask.cpu(), global_attn.cpu()
                )
                model.to(device)
                return preds, logs
            raise

    # ------------------------------------------------------------------
    # Pass 1: NER with CLS-only global attention
    # ------------------------------------------------------------------
    global_attn_1 = torch.zeros_like(input_ids)
    global_attn_1[:, 0] = 1

    pass1_preds, pass1_logits = _ner_with_mps_fallback(global_attn_1)
    pass1_spans = decode_bio_spans(pass1_preds, attention_mask[0], offset_mapping)
    diag["pass1_n_spans"] = len(pass1_spans)

    # ------------------------------------------------------------------
    # Pass 2: optional NER refinement with entity-aware global attention
    # ------------------------------------------------------------------
    if ner_mode == "two-pass" and len(pass1_spans) > 0:
        high_conf_global, n_conf = _confident_global_mask(
            pass1_spans, pass1_logits, ner_confidence, seq_len, device
        )
        diag["pass1_n_high_conf_tokens"] = n_conf

        if n_conf > 0:
            global_attn_2 = global_attn_1.clone()
            global_attn_2[0] = global_attn_2[0] | high_conf_global

            pass2_preds, _ = _ner_with_mps_fallback(global_attn_2)
            final_preds = pass2_preds
            diag["used_pass2_ner"] = True
        else:
            # All Pass 1 spans were below threshold — Pass 2 would be identical
            # to Pass 1 (same global mask), so skip and use Pass 1.
            final_preds = pass1_preds
            diag["pass1_zero_high_conf"] = True
    else:
        final_preds = pass1_preds
        if len(pass1_spans) == 0:
            diag["pass1_zero_high_conf"] = True

    pred_spans = decode_bio_spans(final_preds, attention_mask[0], offset_mapping)

    # Attach text and a default null sentiment to every span
    for s in pred_spans:
        s["text"] = text[s["char_start"]:s["char_end"]]
        s["pred_sentiment"] = None

    # Filter to sentiment-bearing types
    sent_indices = [i for i, s in enumerate(pred_spans) if s["type"] in SENTIMENT_ENTITY_TYPES]
    if not sent_indices:
        return pred_spans, diag

    sent_spans = [pred_spans[i] for i in sent_indices]
    entity_masks = build_entity_masks_for_spans(sent_spans, seq_len, device)

    # ------------------------------------------------------------------
    # Pass 3: sentiment with entity-aware global attention
    # ------------------------------------------------------------------
    global_attn_3 = torch.zeros_like(input_ids)
    global_attn_3[:, 0] = 1
    ep = entity_masks.sum(dim=0).clamp(0, 1).long()
    global_attn_3[0] = global_attn_3[0] | ep

    def _run_sentiment(input_ids, attention_mask, global_attn_3, entity_masks):
        encoder_output_3 = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attn_3,
        )
        # Batch all sentiment entities through the head at once
        n_sent = len(sent_spans)
        encoder_output_tiled = encoder_output_3.expand(n_sent, -1, -1)
        scores = model.sentiment_head(encoder_output_tiled, entity_masks)
        return scores

    try:
        scores = _run_sentiment(input_ids, attention_mask, global_attn_3, entity_masks)
    except RuntimeError as e:
        if "mps" in str(e).lower() and device.type != "cpu":
            model.cpu()
            scores = _run_sentiment(
                input_ids.cpu(), attention_mask.cpu(),
                global_attn_3.cpu(), entity_masks.cpu(),
            )
            model.to(device)
        else:
            raise

    for local_i, original_i in enumerate(sent_indices):
        pred_spans[original_i]["pred_sentiment"] = float(scores[local_i].item())

    return pred_spans, diag


# ---------------------------------------------------------------------------
# Matching predicted spans -> gold entities
# ---------------------------------------------------------------------------
def char_iou(s1_start, s1_end, s2_start, s2_end):
    inter = max(0, min(s1_end, s2_end) - max(s1_start, s2_start))
    if inter == 0:
        return 0.0
    union = max(s1_end, s2_end) - min(s1_start, s2_start)
    return inter / max(union, 1)


def match_predictions_to_gold(pred_spans, gold_entities, iou_threshold=0.5):
    """Greedy span-level matching with type constraint.

    A predicted span matches a gold mention iff:
        - same type
        - char_IoU >= iou_threshold
        - the gold mention isn't already matched by a higher-IoU prediction

    Returns:
        per_type_tp, per_type_fp, per_type_fn (dicts)
        gold_to_preds: dict[gold_entity_idx] -> list of pred_span_indices
    """
    # Flatten gold mentions -> (gold_entity_idx, mention_idx, char_start, char_end, type)
    gold_mentions = []
    for g_idx, gold in enumerate(gold_entities):
        gold_type = gold.get("type", "")
        for m_idx, mention in enumerate(gold.get("ner_mentions", [])):
            cs = mention.get("start_char")
            ce = mention.get("end_char")
            if cs is None or ce is None:
                continue
            gold_mentions.append((g_idx, m_idx, cs, ce, gold_type))

    gold_matched = [False] * len(gold_mentions)
    per_type_tp = defaultdict(int)
    per_type_fp = defaultdict(int)
    per_type_fn = defaultdict(int)
    gold_to_preds = defaultdict(list)

    # Sort predictions by char_start for determinism
    pred_order = sorted(range(len(pred_spans)), key=lambda i: pred_spans[i]["char_start"])

    for p_idx in pred_order:
        p = pred_spans[p_idx]
        p_type = p["type"]
        best_iou = 0.0
        best_gm_idx = None
        for gm_idx, (g_idx, m_idx, gcs, gce, gtype) in enumerate(gold_mentions):
            if gold_matched[gm_idx] or gtype != p_type:
                continue
            iou = char_iou(p["char_start"], p["char_end"], gcs, gce)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gm_idx = gm_idx

        if best_gm_idx is not None:
            gold_matched[best_gm_idx] = True
            g_idx = gold_mentions[best_gm_idx][0]
            gold_to_preds[g_idx].append(p_idx)
            pred_spans[p_idx]["match_iou"] = best_iou
            per_type_tp[p_type] += 1
        else:
            pred_spans[p_idx]["match_iou"] = 0.0
            per_type_fp[p_type] += 1

    for gm_idx, (_, _, _, _, gtype) in enumerate(gold_mentions):
        if not gold_matched[gm_idx]:
            per_type_fn[gtype] += 1

    return {
        "per_type_tp": dict(per_type_tp),
        "per_type_fp": dict(per_type_fp),
        "per_type_fn": dict(per_type_fn),
        "gold_to_preds": dict(gold_to_preds),
        "n_gold_mentions": len(gold_mentions),
        "n_pred_spans": len(pred_spans),
    }


def aggregate_per_gold_entity(pred_spans, gold_entities, gold_to_preds, weight_by_iou=False):
    """Aggregate predicted sentiments per gold canonical entity.

    Args:
        weight_by_iou: If True, weight each matched span's sentiment by its
            char-IoU with the gold mention. Default is unweighted mean-pool.

    Returns:
        results: list of dicts {canonical_id, type, gold_sentiment, agg_pred_sentiment, n_matched}
        coverage: {n_total_gold_with_sentiment, n_covered_gold}
    """
    results = []
    n_total = 0
    n_covered = 0

    for g_idx, gold in enumerate(gold_entities):
        gold_type = gold.get("type", "")
        gold_sent = gold.get("sentiment_score")
        if gold_type not in SENTIMENT_ENTITY_TYPES or gold_sent is None:
            continue
        n_total += 1

        matched_preds = [pred_spans[i] for i in gold_to_preds.get(g_idx, [])
                         if pred_spans[i].get("pred_sentiment") is not None]
        if not matched_preds:
            continue
        n_covered += 1

        if weight_by_iou:
            weights = [p.get("match_iou", 1.0) for p in matched_preds]
            total_weight = sum(weights)
            agg_pred = sum(p["pred_sentiment"] * w for p, w in zip(matched_preds, weights)) / total_weight if total_weight > 0 else 0.0
        else:
            agg_pred = sum(p["pred_sentiment"] for p in matched_preds) / len(matched_preds)

        cid = gold.get("canonical_id")
        if isinstance(cid, list):
            cid = cid[0] if cid else None

        results.append({
            "canonical_id": cid,
            "type": gold_type,
            "gold_sentiment": float(gold_sent),
            "agg_pred_sentiment": float(agg_pred),
            "n_matched_preds": len(matched_preds),
        })

    return results, {"n_total_gold_with_sentiment": n_total, "n_covered_gold": n_covered}


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def bucket_of(score):
    for n, lo, hi in SENTIMENT_BUCKETS:
        if lo <= score < hi:
            return n
    return "very_positive" if score >= 1.0 else "very_negative"


def compute_aggregate_metrics(per_entity_results, match_stats_list, n_articles, elapsed):
    # NER counts
    per_type_tp = defaultdict(int)
    per_type_fp = defaultdict(int)
    per_type_fn = defaultdict(int)
    for s in match_stats_list:
        for k, v in s["per_type_tp"].items(): per_type_tp[k] += v
        for k, v in s["per_type_fp"].items(): per_type_fp[k] += v
        for k, v in s["per_type_fn"].items(): per_type_fn[k] += v

    all_types = sorted(set(per_type_tp) | set(per_type_fp) | set(per_type_fn))
    per_type_ner = {}
    for t in all_types:
        tp, fp, fn = per_type_tp[t], per_type_fp[t], per_type_fn[t]
        p, r, f = f1_from_counts(tp, fp, fn)
        per_type_ner[t] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}

    overall_tp = sum(per_type_tp.values())
    overall_fp = sum(per_type_fp.values())
    overall_fn = sum(per_type_fn.values())
    op, oar, of = f1_from_counts(overall_tp, overall_fp, overall_fn)

    # Sentiment-bearing types only (overall)
    sent_tp = sum(per_type_tp[t] for t in SENTIMENT_ENTITY_TYPES if t in per_type_tp)
    sent_fp = sum(per_type_fp[t] for t in SENTIMENT_ENTITY_TYPES if t in per_type_fp)
    sent_fn = sum(per_type_fn[t] for t in SENTIMENT_ENTITY_TYPES if t in per_type_fn)
    sp, sr, sf = f1_from_counts(sent_tp, sent_fp, sent_fn)

    # Coverage
    n_total_gold = sum(s.get("n_total_gold_with_sentiment", 0) for s in match_stats_list)
    n_covered_gold = sum(s.get("n_covered_gold", 0) for s in match_stats_list)
    coverage_rate = n_covered_gold / n_total_gold if n_total_gold > 0 else 0.0

    # Sentiment metrics on covered entities
    sent_metrics = {}
    bucket_confusion = defaultdict(lambda: defaultdict(int))
    if per_entity_results:
        gold = torch.tensor([r["gold_sentiment"] for r in per_entity_results])
        pred = torch.tensor([r["agg_pred_sentiment"] for r in per_entity_results])
        mse = ((pred - gold) ** 2).mean().item()
        mae = (pred - gold).abs().mean().item()
        if pred.numel() > 1 and pred.std().item() > 0 and gold.std().item() > 0:
            vx = pred - pred.mean()
            vy = gold - gold.mean()
            denom = vx.norm().item() * vy.norm().item()
            pearson_r = (vx * vy).sum().item() / denom if denom > 0 else 0.0
        else:
            pearson_r = 0.0
        sent_metrics = {
            "n": len(per_entity_results),
            "mse": mse,
            "mae": mae,
            "pearson_r": pearson_r,
            "pred_min": pred.min().item(),
            "pred_max": pred.max().item(),
            "pred_mean": pred.mean().item(),
            "pred_std": pred.std().item(),
            "gold_min": gold.min().item(),
            "gold_max": gold.max().item(),
            "gold_mean": gold.mean().item(),
            "gold_std": gold.std().item(),
        }
        for r in per_entity_results:
            bucket_confusion[bucket_of(r["gold_sentiment"])][bucket_of(r["agg_pred_sentiment"])] += 1

    # Per-type sentiment metrics
    per_type_sentiment = {}
    by_type = defaultdict(lambda: {"gold": [], "pred": []})
    for r in per_entity_results:
        by_type[r["type"]]["gold"].append(r["gold_sentiment"])
        by_type[r["type"]]["pred"].append(r["agg_pred_sentiment"])
    for t, d in by_type.items():
        if not d["gold"]:
            continue
        g = torch.tensor(d["gold"])
        p = torch.tensor(d["pred"])
        mse = ((p - g) ** 2).mean().item()
        mae = (p - g).abs().mean().item()
        if p.numel() > 1 and p.std().item() > 0 and g.std().item() > 0:
            vx = p - p.mean()
            vy = g - g.mean()
            denom = vx.norm().item() * vy.norm().item()
            pr = (vx * vy).sum().item() / denom if denom > 0 else 0.0
        else:
            pr = 0.0
        per_type_sentiment[t] = {"n": len(d["gold"]), "mse": mse, "mae": mae, "pearson_r": pr}

    # Joint accuracy (matched + sentiment within tolerance)
    joint_acc = {}
    for tol in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if per_entity_results:
            correct = sum(1 for r in per_entity_results
                          if abs(r["gold_sentiment"] - r["agg_pred_sentiment"]) <= tol)
            joint_acc[f"tol_{tol}"] = correct / len(per_entity_results)
        else:
            joint_acc[f"tol_{tol}"] = 0.0

    # Bucket accuracy
    diag = sum(bucket_confusion.get(b, {}).get(b, 0) for b in BUCKET_NAMES)
    total_b = sum(sum(v.values()) for v in bucket_confusion.values())
    bucket_acc = diag / total_b if total_b > 0 else 0.0
    adj = 0
    for i, tb in enumerate(BUCKET_NAMES):
        for j, pb in enumerate(BUCKET_NAMES):
            if abs(i - j) <= 1:
                adj += bucket_confusion.get(tb, {}).get(pb, 0)
    adj_acc = adj / total_b if total_b > 0 else 0.0

    # Distribution of matched predictions per gold entity
    matched_preds_dist = defaultdict(int)
    for r in per_entity_results:
        matched_preds_dist[r["n_matched_preds"]] += 1

    return {
        "n_articles": n_articles,
        "elapsed_seconds": elapsed,
        "ner_metrics": {
            "overall_all_types": {
                "precision": op, "recall": oar, "f1": of,
                "tp": overall_tp, "fp": overall_fp, "fn": overall_fn,
            },
            "overall_sentiment_types": {
                "precision": sp, "recall": sr, "f1": sf,
                "tp": sent_tp, "fp": sent_fp, "fn": sent_fn,
            },
            "per_type": per_type_ner,
        },
        "coverage": {
            "n_total_gold_with_sentiment": n_total_gold,
            "n_covered": n_covered_gold,
            "coverage_rate": coverage_rate,
        },
        "sentiment_on_covered": sent_metrics,
        "per_type_sentiment": per_type_sentiment,
        "joint_accuracy": joint_acc,
        "bucket_confusion": {tb: dict(pds) for tb, pds in bucket_confusion.items()},
        "bucket_metrics": {"exact_acc": bucket_acc, "adjacent_acc": adj_acc},
        "matched_preds_distribution": dict(matched_preds_dist),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_summary(path, metrics, meta):
    L = []
    L.append("=" * 80)
    L.append(f"  END-TO-END PIPELINE EVALUATION")
    L.append(f"  Checkpoint: {meta['checkpoint']}")
    L.append(f"  Benchmark : {meta['benchmark']}")
    L.append(f"  Articles  : {metrics['n_articles']:,}    "
             f"Elapsed: {metrics['elapsed_seconds']:.1f}s    "
             f"IoU>={meta['iou_threshold']}")
    if meta.get("weighted_aggregation"):
        L.append("  Aggregation: IoU-weighted mean")
    else:
        L.append("  Aggregation: unweighted mean-pool")
    L.append(f"  NER mode  : {meta.get('ner_mode', 'unknown')}  "
             f"(confidence_threshold={meta.get('ner_confidence', 'n/a')})")
    if meta.get("ner_mode") == "single-pass":
        L.append("  NOTE: single-pass NER uses CLS-only global attention (train/test mismatch)")
    L.append("=" * 80)
    L.append("")

    diag = meta.get("ner_diagnostics", {})
    if diag:
        L.append("-" * 60)
        L.append("  NER PIPELINE DIAGNOSTICS")
        L.append("-" * 60)
        L.append(f"  Articles where Pass 1 produced 0 confident spans: "
                 f"{diag.get('n_articles_pass1_zero_confident', 0):,}  "
                 f"({diag.get('frac_articles_pass1_zero_confident', 0)*100:.1f}%)")
        L.append(f"  Articles where Pass 2 NER ran                    : "
                 f"{diag.get('n_articles_used_pass2_ner', 0):,}  "
                 f"({diag.get('frac_articles_used_pass2_ner', 0)*100:.1f}%)")
        L.append(f"  Avg Pass 1 spans per article                     : "
                 f"{diag.get('avg_pass1_spans_per_article', 0):.2f}")
        L.append(f"  Avg Pass 1 high-confidence tokens per article    : "
                 f"{diag.get('avg_pass1_high_conf_tokens_per_article', 0):.2f}")
        L.append("")

    L.append("-" * 60)
    L.append("  NER (SPAN-LEVEL, all entity types)")
    L.append("-" * 60)
    o = metrics["ner_metrics"]["overall_all_types"]
    L.append(f"  Precision: {o['precision']:.4f}   Recall: {o['recall']:.4f}   F1: {o['f1']:.4f}")
    L.append(f"  TP={o['tp']:,}   FP={o['fp']:,}   FN={o['fn']:,}")
    L.append("")
    L.append("  NER (sentiment-bearing types only: ORG/TICKER/PERSON/COMPANY)")
    s = metrics["ner_metrics"]["overall_sentiment_types"]
    L.append(f"  Precision: {s['precision']:.4f}   Recall: {s['recall']:.4f}   F1: {s['f1']:.4f}")
    L.append(f"  TP={s['tp']:,}   FP={s['fp']:,}   FN={s['fn']:,}")
    L.append("")
    L.append(f"  {'Type':<10} {'Prec':>8} {'Recall':>8} {'F1':>8}  {'TP':>8} {'FP':>8} {'FN':>8}")
    L.append(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}  {'-'*8} {'-'*8} {'-'*8}")
    for t, m in sorted(metrics["ner_metrics"]["per_type"].items()):
        L.append(f"  {t:<10} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}  "
                 f"{m['tp']:>8,} {m['fp']:>8,} {m['fn']:>8,}")
    L.append("")

    L.append("-" * 60)
    L.append("  ENTITY COVERAGE (sentiment-bearing gold entities)")
    L.append("-" * 60)
    c = metrics["coverage"]
    L.append(f"  Coverage rate: {c['coverage_rate']:.4f}")
    L.append(f"  Covered      : {c['n_covered']:,} / {c['n_total_gold_with_sentiment']:,}")
    L.append("")

    L.append("-" * 60)
    L.append("  SENTIMENT QUALITY (on covered gold entities)")
    L.append("-" * 60)
    sm = metrics["sentiment_on_covered"]
    if sm:
        L.append(f"  N entities    : {sm['n']:,}")
        L.append(f"  MSE           : {sm['mse']:.6f}")
        L.append(f"  MAE           : {sm['mae']:.6f}")
        L.append(f"  Pearson r     : {sm['pearson_r']:.6f}")
        L.append(f"  Pred range    : [{sm['pred_min']:.4f}, {sm['pred_max']:.4f}]   "
                 f"mean={sm['pred_mean']:.4f}   std={sm['pred_std']:.4f}")
        L.append(f"  Gold range    : [{sm['gold_min']:.4f}, {sm['gold_max']:.4f}]   "
                 f"mean={sm['gold_mean']:.4f}   std={sm['gold_std']:.4f}")
    L.append("")

    L.append("-" * 60)
    L.append("  PER-TYPE SENTIMENT (on covered)")
    L.append("-" * 60)
    L.append(f"  {'Type':<10} {'N':>8} {'MSE':>10} {'MAE':>10} {'Corr':>10}")
    L.append(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for t, m in sorted(metrics["per_type_sentiment"].items()):
        L.append(f"  {t:<10} {m['n']:>8,} {m['mse']:>10.6f} {m['mae']:>10.6f} {m['pearson_r']:>10.6f}")
    L.append("")

    L.append("-" * 60)
    L.append("  JOINT ACCURACY (matched span + sentiment within tolerance)")
    L.append("-" * 60)
    for k in sorted(metrics["joint_accuracy"].keys()):
        tol = k.replace("tol_", "")
        L.append(f"  |pred - gold| <= {tol}: {metrics['joint_accuracy'][k]:.4f}")
    L.append("")

    L.append("-" * 60)
    L.append("  SENTIMENT BUCKET CONFUSION  (rows=true, cols=predicted)")
    L.append("-" * 60)
    header = f"  {'True \\ Pred':<16}"
    for bn in BUCKET_NAMES:
        header += f" {bn:>14}"
    header += f" {'Total':>10}"
    L.append(header)
    L.append(f"  {'-'*16}" + (f" {'-'*14}" * len(BUCKET_NAMES)) + f" {'-'*10}")
    grand = 0
    for tb in BUCKET_NAMES:
        row_sum = sum(metrics["bucket_confusion"].get(tb, {}).get(pb, 0) for pb in BUCKET_NAMES)
        grand += row_sum
        row = f"  {tb:<16}"
        for pb in BUCKET_NAMES:
            row += f" {metrics['bucket_confusion'].get(tb, {}).get(pb, 0):>14,}"
        row += f" {row_sum:>10,}"
        L.append(row)
    bm = metrics["bucket_metrics"]
    L.append("")
    L.append(f"  Exact bucket accuracy   : {bm['exact_acc']:.4f}")
    L.append(f"  Adjacent bucket accuracy: {bm['adjacent_acc']:.4f}")
    L.append("")

    L.append("-" * 60)
    L.append("  MATCHED PREDICTIONS PER GOLD ENTITY")
    L.append("-" * 60)
    mpd = metrics.get("matched_preds_distribution", {})
    if mpd:
        for k in sorted(mpd.keys()):
            L.append(f"  {k} matched span(s): {mpd[k]:,}")
    L.append("")
    L.append("=" * 80)

    with open(path, "w") as f:
        f.write("\n".join(L))


def print_short_summary(metrics, meta):
    print()
    print("=" * 80)
    print(f"  E2E EVALUATION  ner_mode={meta.get('ner_mode', 'unknown')}")
    if meta.get("weighted_aggregation"):
        print("  Aggregation: IoU-weighted")
    print("=" * 80)
    o = metrics["ner_metrics"]["overall_all_types"]
    s = metrics["ner_metrics"]["overall_sentiment_types"]
    print(f"  NER F1 (all types)           : {o['f1']:.4f}  (P={o['precision']:.4f} R={o['recall']:.4f})")
    print(f"  NER F1 (sentiment types only): {s['f1']:.4f}  (P={s['precision']:.4f} R={s['recall']:.4f})")
    c = metrics["coverage"]
    print(f"  Entity coverage              : {c['coverage_rate']:.4f}  ({c['n_covered']:,}/{c['n_total_gold_with_sentiment']:,})")
    sm = metrics["sentiment_on_covered"]
    if sm:
        print(f"  Sentiment MSE / MAE / Corr   : {sm['mse']:.4f} / {sm['mae']:.4f} / {sm['pearson_r']:.4f}")
    j = metrics["joint_accuracy"]
    print(f"  Joint acc tol=0.2            : {j.get('tol_0.2', 0):.4f}")
    print(f"  Joint acc tol=0.4            : {j.get('tol_0.4', 0):.4f}")
    bm = metrics["bucket_metrics"]
    print(f"  Bucket exact / adjacent      : {bm['exact_acc']:.4f} / {bm['adjacent_acc']:.4f}")
    mpd = metrics.get("matched_preds_distribution", {})
    if mpd:
        print(f"  Matched spans/entity         : mostly {max(mpd, key=mpd.get)} (max={max(mpd.values()):,})")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--benchmark", type=str, default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save per-article predictions to JSONL (large file).")
    parser.add_argument("--local-output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR / "local"),
                        help="Local dir to write outputs first (Colab pattern).")
    parser.add_argument("--terminate-runtime", action="store_true",
                        help="Auto-terminate Colab runtime when done.")
    parser.add_argument("--weighted-aggregation", action="store_true",
                        help="Weight sentiment aggregation by char-IoU of each matched span.")
    parser.add_argument("--ner-mode", type=str, default="two-pass",
                        choices=["single-pass", "two-pass"],
                        help="single-pass: NER with CLS-only global (train/test mismatch). "
                             "two-pass: refine NER with entity-aware global from Pass 1.")
    parser.add_argument("--ner-confidence", type=float, default=0.7,
                        help="Pass 1 softmax max-prob threshold for seeding Pass 2 global "
                             "attention. Tokens below this are not promoted to global.")
    parser.add_argument("--inference-batch-size", type=int, default=1,
                        help="Articles per encoder forward in single-pass mode. "
                             "Default 1 uses the legacy per-article path. "
                             "On a 95GB GPU at seq_len=2048, batch=8 is safe; "
                             "batch=16 fits with no_grad. Two-pass mode ignores "
                             "this (always per-article).")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Output paths (local-first)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    drive_output_dir = Path(args.output_dir)
    drive_output_dir.mkdir(parents=True, exist_ok=True)
    local_output_dir = Path(args.local_output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)

    local_metrics = local_output_dir / f"e2e_metrics_{timestamp}.json"
    local_summary = local_output_dir / f"e2e_summary_{timestamp}.txt"
    local_preds   = local_output_dir / f"e2e_predictions_{timestamp}.jsonl"
    local_log     = local_output_dir / f"e2e_eval_{timestamp}.log"

    # File-handler log (matches Colab pattern from train_stage3.py)
    fh = logging.FileHandler(str(local_log))
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 60)
    logger.info("END-TO-END PIPELINE EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Checkpoint    : {args.checkpoint}")
    logger.info(f"Benchmark     : {args.benchmark}")
    logger.info(f"IoU threshold : {args.iou_threshold}")
    logger.info(f"Max length    : {args.max_length}")
    logger.info(f"Local outputs : {local_output_dir}")
    logger.info(f"Drive outputs : {drive_output_dir}")
    logger.info(f"Weighted aggregation: {args.weighted_aggregation}")
    logger.info(f"NER mode      : {args.ner_mode}  (confidence={args.ner_confidence})")

    # Load model
    model, epoch, val_metrics = load_model(args.checkpoint, device)
    logger.info(f"Loaded model (epoch={epoch})")
    if val_metrics:
        logger.info(f"  Saved val metrics: corr={val_metrics.get('sentiment_corr', 'n/a')}, "
                    f"mse={val_metrics.get('sentiment_mse', 'n/a')}")

    tokenizer = LongformerTokenizerFast.from_pretrained(ENCODER_NAME)

    # Load benchmark
    logger.info(f"Loading benchmark: {args.benchmark}")
    articles = []
    skipped = 0
    with open(args.benchmark) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
                continue
    logger.info(f"Loaded {len(articles)} articles ({skipped} skipped)")

    if args.max_samples > 0 and len(articles) > args.max_samples:
        import random
        random.seed(42)
        articles = random.sample(articles, args.max_samples)
        logger.info(f"Sub-sampled to {len(articles)} (seed=42)")

    # Run evaluation
    all_per_entity = []
    all_match_stats = []
    n_failed = 0
    failure_reasons = defaultdict(int)

    # NER mode diagnostics (per-article)
    diag_n_pass1_zero = 0          # articles with no Pass 1 spans or none confident
    diag_n_pass2_used = 0          # articles where Pass 2 NER actually ran
    diag_total_pass1_spans = 0
    diag_total_pass1_high_conf = 0

    # Use batched single-pass when conditions match (no MPS quirks, single-pass mode, batch > 1)
    use_batched = (
        args.ner_mode == "single-pass"
        and args.inference_batch_size > 1
        and device.type != "mps"
    )
    logger.info(f"Inference path: {'batched single-pass (mini_batch=' + str(args.inference_batch_size) + ')' if use_batched else 'per-article'}")

    t_start = time.time()
    pred_file_ctx = open(local_preds, "w") if args.save_predictions else None
    with pred_file_ctx if pred_file_ctx else contextlib.nullcontext() as pred_file:
        # ------------------------------------------------------------------
        # Batched path (single-pass NER, mini_batch_size > 1)
        # ------------------------------------------------------------------
        if use_batched:
            mb = args.inference_batch_size
            for batch_start in range(0, len(articles), mb):
                batch_articles = articles[batch_start:batch_start + mb]
                try:
                    batch_results = predict_articles_batched(
                        model, tokenizer, batch_articles, device,
                        max_length=args.max_length, mini_batch_size=mb,
                    )
                except (RuntimeError, IndexError, KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Batch starting at {batch_start}: error - {type(e).__name__}: {e}. Falling back to per-article for this batch.")
                    batch_results = []
                    for art in batch_articles:
                        try:
                            spans, diag = predict_article(
                                model, tokenizer, art.get("text", ""), device,
                                args.max_length, ner_mode="single-pass",
                                ner_confidence=args.ner_confidence,
                            )
                            batch_results.append((spans, diag))
                        except Exception as e2:
                            batch_results.append(None)
                            failure_reasons[type(e2).__name__] += 1

                for local_i, result in enumerate(batch_results):
                    global_i = batch_start + local_i
                    article = batch_articles[local_i]
                    article_id = article.get("id", f"unknown_{global_i}")

                    if result is None:
                        n_failed += 1
                        continue
                    pred_spans, art_diag = result
                    if art_diag.get("empty_text"):
                        n_failed += 1
                        failure_reasons["empty_text"] += 1
                        continue

                    gold_entities = article.get("entities", [])
                    diag_total_pass1_spans += art_diag["pass1_n_spans"]
                    diag_total_pass1_high_conf += art_diag["pass1_n_high_conf_tokens"]
                    if art_diag["pass1_zero_high_conf"]:
                        diag_n_pass1_zero += 1
                    if art_diag["used_pass2_ner"]:
                        diag_n_pass2_used += 1

                    match_stats = match_predictions_to_gold(pred_spans, gold_entities, args.iou_threshold)
                    per_entity, coverage = aggregate_per_gold_entity(
                        pred_spans, gold_entities, match_stats["gold_to_preds"],
                        weight_by_iou=args.weighted_aggregation,
                    )
                    match_stats["n_total_gold_with_sentiment"] = coverage["n_total_gold_with_sentiment"]
                    match_stats["n_covered_gold"] = coverage["n_covered_gold"]
                    all_per_entity.extend(per_entity)
                    all_match_stats.append(match_stats)

                    if pred_file:
                        pred_file.write(json.dumps({
                            "id": article_id,
                            "predicted_spans": pred_spans,
                            "per_entity_results": per_entity,
                            "ner_counts": {
                                "tp": match_stats["per_type_tp"],
                                "fp": match_stats["per_type_fp"],
                                "fn": match_stats["per_type_fn"],
                            },
                            "coverage": {
                                "n_total_gold_with_sentiment": coverage["n_total_gold_with_sentiment"],
                                "n_covered_gold": coverage["n_covered_gold"],
                            },
                            "ner_diagnostics": art_diag,
                        }) + "\n")

                done = min(batch_start + mb, len(articles))
                if done % args.log_every < mb or done == len(articles):
                    elapsed = time.time() - t_start
                    rate = done / elapsed
                    eta = (len(articles) - done) / rate if rate > 0 else 0
                    logger.info(f"  [{done}/{len(articles)}]  {rate:.2f} art/s  "
                                f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
            # End batched path — skip per-article block below
            articles_iter = []
        else:
            articles_iter = list(enumerate(articles))

        # ------------------------------------------------------------------
        # Per-article path (legacy / 2-pass / batch_size=1 / MPS)
        # ------------------------------------------------------------------
        for i, article in articles_iter:
            text = article.get("text", "")
            if not text.strip():
                n_failed += 1
                failure_reasons["empty_text"] += 1
                continue

            gold_entities = article.get("entities", [])
            article_id = article.get("id", f"unknown_{i}")

            try:
                pred_spans, art_diag = predict_article(
                    model, tokenizer, text, device, args.max_length,
                    ner_mode=args.ner_mode, ner_confidence=args.ner_confidence,
                )
            except (RuntimeError, IndexError, KeyError, ValueError, TypeError) as e:
                logger.warning(f"Article {i} ({article_id}): inference error - {type(e).__name__}: {e}")
                n_failed += 1
                failure_reasons[type(e).__name__] += 1
                continue

            # Aggregate per-article NER-mode diagnostics
            diag_total_pass1_spans += art_diag["pass1_n_spans"]
            diag_total_pass1_high_conf += art_diag["pass1_n_high_conf_tokens"]
            if art_diag["pass1_zero_high_conf"]:
                diag_n_pass1_zero += 1
            if art_diag["used_pass2_ner"]:
                diag_n_pass2_used += 1

            match_stats = match_predictions_to_gold(pred_spans, gold_entities, args.iou_threshold)
            per_entity, coverage = aggregate_per_gold_entity(
                pred_spans, gold_entities, match_stats["gold_to_preds"],
                weight_by_iou=args.weighted_aggregation,
            )
            match_stats["n_total_gold_with_sentiment"] = coverage["n_total_gold_with_sentiment"]
            match_stats["n_covered_gold"] = coverage["n_covered_gold"]

            all_per_entity.extend(per_entity)
            all_match_stats.append(match_stats)

            if pred_file:
                pred_file.write(json.dumps({
                    "id": article_id,
                    "predicted_spans": pred_spans,
                    "per_entity_results": per_entity,
                    "ner_counts": {
                        "tp": match_stats["per_type_tp"],
                        "fp": match_stats["per_type_fp"],
                        "fn": match_stats["per_type_fn"],
                    },
                    "coverage": {
                        "n_total_gold_with_sentiment": coverage["n_total_gold_with_sentiment"],
                        "n_covered_gold": coverage["n_covered_gold"],
                    },
                    "ner_diagnostics": art_diag,
                }) + "\n")

            if (i + 1) % args.log_every == 0 or (i + 1) == len(articles):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(articles) - i - 1) / rate if rate > 0 else 0
                logger.info(f"  [{i+1}/{len(articles)}]  {rate:.2f} art/s  "
                            f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    if failure_reasons:
        logger.info(f"Failure breakdown: {dict(failure_reasons)}")

    elapsed_total = time.time() - t_start
    logger.info(f"Inference complete. {n_failed} articles failed/skipped.")

    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(
        all_per_entity, all_match_stats, len(articles) - n_failed, elapsed_total,
    )
    n_succ = max(len(articles) - n_failed, 1)
    meta = {
        "checkpoint": str(args.checkpoint),
        "benchmark": str(args.benchmark),
        "checkpoint_epoch": epoch,
        "iou_threshold": args.iou_threshold,
        "weighted_aggregation": args.weighted_aggregation,
        "ner_mode": args.ner_mode,
        "ner_confidence": args.ner_confidence,
        "timestamp": timestamp,
        "device": str(device),
        "max_length": args.max_length,
        "n_articles_attempted": len(articles),
        "n_articles_succeeded": len(articles) - n_failed,
        "failure_reasons": dict(failure_reasons),
        "ner_diagnostics": {
            "n_articles_pass1_zero_confident": diag_n_pass1_zero,
            "frac_articles_pass1_zero_confident": diag_n_pass1_zero / n_succ,
            "n_articles_used_pass2_ner": diag_n_pass2_used,
            "frac_articles_used_pass2_ner": diag_n_pass2_used / n_succ,
            "avg_pass1_spans_per_article": diag_total_pass1_spans / n_succ,
            "avg_pass1_high_conf_tokens_per_article": diag_total_pass1_high_conf / n_succ,
        },
    }
    full_output = {"meta": meta, **metrics}

    # Save locally first
    with open(local_metrics, "w") as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"Saved metrics: {local_metrics}")

    write_summary(local_summary, metrics, meta)
    logger.info(f"Saved summary: {local_summary}")

    # Sync to Drive
    logger.info("Syncing outputs to Drive...")
    for src in [local_metrics, local_summary, local_log] + ([local_preds] if args.save_predictions else []):
        try:
            shutil.copy2(str(src), str(drive_output_dir / src.name))
            logger.info(f"  {src.name} -> Drive OK")
        except Exception as e:
            logger.warning(f"  {src.name} -> Drive FAILED: {e}")

    # Print short summary to console
    print_short_summary(metrics, meta)

    # Optional: terminate runtime
    if args.terminate_runtime:
        try:
            from google.colab import drive
            drive.flush_and_unmount()
            logger.info("Drive flushed.")
            drive.mount("/content/drive")
            logger.info("Drive remounted.")
            time.sleep(5)
        except Exception as e:
            logger.warning(f"Drive flush/remount issue: {e}")

        logger.info("Final checkpoint verification:")
        try:
            if drive_output_dir.exists():
                for f in sorted(os.listdir(str(drive_output_dir))):
                    if f.startswith("e2e_"):
                        size = os.path.getsize(str(drive_output_dir / f)) / 1e6
                        logger.info(f"  {f:35s} {size:8.1f} MB")
            else:
                logger.warning(f"  Drive directory not accessible yet: {drive_output_dir}")
                logger.info(f"  Local outputs are safe at {local_output_dir}")
        except Exception as e:
            logger.warning(f"  Verification skipped: {e}")

        try:
            from google.colab import runtime
            logger.info("Terminating runtime...")
            runtime.unassign()
        except Exception:
            pass


if __name__ == "__main__":
    main()
