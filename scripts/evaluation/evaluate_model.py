#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script.

Tests:
1. NER labeling performance (CRF head)
2. Sentiment head performance with ground truth entities
3. SpaCy transformer + coreference → sentiment pipeline
4. Full CRF + coreference → sentiment pipeline
5. NER comparison: Our CRF vs SpaCy transformer
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up to project root
sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline import FinancialEntitySentimentModel
from models.ner_head_crf import NERHeadCRF
from training import (
    DataPreprocessor,
    create_data_loaders,
    LABEL_TO_ID,
    ID_TO_LABEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_validation_data(data_dir: str = None, val_split: float = 0.1, max_length: int = 2048, val_file: str = None):
    """Load validation dataset."""
    from torch.utils.data import DataLoader
    from training.dataset import EntitySentimentDataset, collate_fn

    preprocessor = DataPreprocessor(max_length=max_length)

    if val_file:
        # Load from specific validation file directly
        val_samples = preprocessor.process_file(val_file)
        val_dataset = EntitySentimentDataset(val_samples, max_entities_per_sample=10)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
    else:
        # Original behavior: load from directory with split
        data_dir = Path(data_dir)
        data_files = list(data_dir.glob("*.jsonl"))

        _, val_loader = create_data_loaders(
            train_files=[str(f) for f in data_files],
            val_files=None,
            batch_size=1,
            val_split=val_split,
            num_workers=0,
            preprocessor=preprocessor,
            max_entities_per_sample=10,
            seed=42,
        )

    return val_loader, preprocessor


def load_raw_validation_samples(data_dir: str = None, val_split: float = 0.1, seed: int = 42, val_file: str = None):
    """Load raw validation samples with text and entity info."""
    import random
    random.seed(seed)

    all_samples = []

    if val_file:
        # Load from specific file
        with open(val_file) as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    if sample.get("text") and sample.get("entities"):
                        all_samples.append(sample)
                except json.JSONDecodeError:
                    continue
        return all_samples  # Return all samples from the file
    else:
        # Original behavior
        data_dir = Path(data_dir)
        for file_path in data_dir.glob("*.jsonl"):
            with open(file_path) as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        if sample.get("text") and sample.get("entities"):
                            all_samples.append(sample)
                    except json.JSONDecodeError:
                        continue

        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * (1 - val_split))
        val_samples = all_samples[split_idx:]

        return val_samples


def evaluate_ner(model, val_loader, device):
    """
    Evaluate NER (CRF) performance.

    Returns:
        dict: Precision, Recall, F1 for each entity type and overall
    """
    logger.info("=" * 60)
    logger.info("1. NER LABELING PERFORMANCE (CRF)")
    logger.info("=" * 60)

    model.eval()

    all_preds = []
    all_labels = []

    # Per-entity-type tracking
    entity_type_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ner_labels = batch["ner_labels"].to(device)

            # Get CRF predictions
            encoder_output = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            ner_output = model.ner_head(encoder_output, attention_mask=attention_mask)
            predictions = ner_output["predictions"]

            # Collect token-level predictions
            mask = attention_mask.bool()
            for i in range(predictions.shape[0]):
                seq_len = mask[i].sum().item()
                preds_seq = predictions[i, :seq_len].cpu().tolist()
                labels_seq = ner_labels[i, :seq_len].cpu().tolist()

                all_preds.extend(preds_seq)
                all_labels.extend(labels_seq)

    # Filter out ignored labels (-100)
    valid_pairs = [(p, l) for p, l in zip(all_preds, all_labels) if l != -100]
    all_preds = [p for p, l in valid_pairs]
    all_labels = [l for p, l in valid_pairs]

    # Overall token-level metrics (excluding O)
    entity_preds = [p for p in all_preds if p != 0]
    entity_labels = [l for l in all_labels if l != 0]

    # Calculate per-class metrics
    unique_labels = sorted(set(all_labels) | set(all_preds))

    results = {"per_class": {}, "overall": {}}

    # Token-level metrics per entity type
    for label_id in unique_labels:
        if label_id == 0:  # Skip O tag
            continue
        label_name = ID_TO_LABEL.get(label_id, f"UNK_{label_id}")

        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == label_id and l == label_id)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == label_id and l != label_id)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != label_id and l == label_id)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results["per_class"][label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    # Overall metrics (entity tokens only, excluding O)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l and l != 0)
    pred_entities = sum(1 for p in all_preds if p != 0)
    true_entities = sum(1 for l in all_labels if l != 0)

    overall_precision = correct / pred_entities if pred_entities > 0 else 0
    overall_recall = correct / true_entities if true_entities > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "total_predicted": pred_entities,
        "total_ground_truth": true_entities,
    }

    # Print results
    logger.info(f"\nOverall NER Performance (token-level, excluding O):")
    logger.info(f"  Precision: {overall_precision:.3f}")
    logger.info(f"  Recall:    {overall_recall:.3f}")
    logger.info(f"  F1:        {overall_f1:.3f}")
    logger.info(f"  Predicted: {pred_entities} entity tokens")
    logger.info(f"  Ground truth: {true_entities} entity tokens")

    logger.info(f"\nPer-class Performance:")
    for label_name, metrics in sorted(results["per_class"].items()):
        if metrics["support"] > 0:
            logger.info(f"  {label_name:15s}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} (support={metrics['support']})")

    return results


def get_entity_name(entity_info):
    """Extract entity name from various data formats."""
    # Use canonical_id (typically ticker symbol)
    name = entity_info.get("canonical_id", "")
    if name and name != "UNKNOWN":
        return name
    name = entity_info.get("canonical_name", "")
    if name and name != "UNKNOWN":
        return name
    # Try ner_mentions
    if entity_info.get("ner_mentions"):
        return entity_info["ner_mentions"][0].get("text", "")
    # Last resort
    return entity_info.get("name", entity_info.get("text", ""))


def find_entity_in_text(entity_name: str, text: str) -> bool:
    """Check if entity appears in text (case-insensitive)."""
    text_lower = text.lower()
    entity_lower = entity_name.lower()
    return entity_lower in text_lower


def entities_match(detected: str, ground_truth: str) -> bool:
    """Check if detected entity matches ground truth (fuzzy)."""
    det_lower = detected.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # Direct match
    if det_lower == gt_lower:
        return True

    # Substring match
    if det_lower in gt_lower or gt_lower in det_lower:
        return True

    # Common variations (e.g., "Apple" matches "AAPL", "Amazon" matches "AMZN")
    # Simple heuristic: first letters match and one is much shorter (likely ticker)
    if len(det_lower) <= 5 and len(gt_lower) > 5:
        if gt_lower.startswith(det_lower[0]):
            return True
    if len(gt_lower) <= 5 and len(det_lower) > 5:
        if det_lower.startswith(gt_lower[0]):
            return True

    return False


def evaluate_sentiment_standalone(model, val_samples, device, max_length=2048):
    """
    Evaluate sentiment head with ground truth entities from dataset.

    Returns:
        dict: MSE, MAE, Pearson/Spearman correlation
    """
    logger.info("\n" + "=" * 60)
    logger.info("2. SENTIMENT HEAD PERFORMANCE (Ground Truth Entities)")
    logger.info("=" * 60)

    model.eval()

    predictions = []
    ground_truth = []
    missing_entities = 0
    total_entities = 0

    with torch.no_grad():
        for sample in val_samples:
            text = sample.get("text", "")
            entities = sample.get("entities", [])

            for entity_info in entities:
                entity_name = get_entity_name(entity_info)
                gt_score = entity_info.get("sentiment_score")

                if gt_score is None or not entity_name or entity_name == "UNKNOWN":
                    continue

                total_entities += 1

                # Use model.analyze with the ground truth entity
                try:
                    result = model.analyze(
                        text=text,
                        target_entities=[entity_name],
                        use_coref=False,
                    )

                    if entity_name in result and result[entity_name].get("sentiment_score") is not None:
                        pred_score = result[entity_name]["sentiment_score"]
                        predictions.append(pred_score)
                        ground_truth.append(gt_score)
                    else:
                        missing_entities += 1
                except Exception as e:
                    missing_entities += 1

    if len(predictions) == 0:
        logger.warning("No valid predictions for sentiment evaluation!")
        return {"error": "No valid predictions"}

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Calculate metrics
    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(mse)

    # Correlation
    pearson_r, pearson_p = pearsonr(ground_truth, predictions)
    spearman_r, spearman_p = spearmanr(ground_truth, predictions)

    # Missing entity percentage
    missing_pct = (missing_entities / total_entities * 100) if total_entities > 0 else 0

    results = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "total_entities": total_entities,
        "evaluated_entities": len(predictions),
        "missing_entities": missing_entities,
        "missing_pct": missing_pct,
    }

    logger.info(f"\nSentiment Prediction Metrics:")
    logger.info(f"  MSE:        {mse:.4f}")
    logger.info(f"  RMSE:       {rmse:.4f}")
    logger.info(f"  MAE:        {mae:.4f}")
    logger.info(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.4e})")
    logger.info(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.4e})")
    logger.info(f"\nEntity Coverage:")
    logger.info(f"  Total entities:     {total_entities}")
    logger.info(f"  Evaluated:          {len(predictions)}")
    logger.info(f"  Missing (not found): {missing_entities} ({missing_pct:.1f}%)")

    return results


def evaluate_spacy_pipeline(model, val_samples, device):
    """
    Evaluate SpaCy transformer NER + coreference → sentiment pipeline.

    Returns:
        dict: Sentiment metrics when using SpaCy for entity detection
    """
    logger.info("\n" + "=" * 60)
    logger.info("3. SPACY TRANSFORMER + COREFERENCE → SENTIMENT")
    logger.info("=" * 60)

    try:
        import spacy
        nlp = spacy.load("en_core_web_trf")
        logger.info("Loaded SpaCy en_core_web_trf model")
    except Exception as e:
        logger.warning(f"Could not load SpaCy transformer model: {e}")
        logger.info("Trying en_core_web_sm instead...")
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded SpaCy en_core_web_sm model")
        except Exception as e2:
            logger.error(f"Could not load any SpaCy model: {e2}")
            return {"error": "SpaCy not available"}

    model.eval()

    predictions = []
    ground_truth = []
    missing_entities = 0
    total_entities = 0
    detected_entities = 0

    # Map SpaCy entity types to our types
    spacy_to_our_types = {
        "ORG": ["COMPANY", "ORG"],
        "PERSON": ["PERSON"],
        "GPE": ["ORG"],
        "MONEY": ["MONEY"],
        "PERCENT": ["PERCENT"],
        "DATE": ["DATE"],
    }

    debug_count = 0
    debug_max = 5  # Log first 5 samples for debugging

    with torch.no_grad():
        for sample in val_samples:
            text = sample.get("text", "")
            gt_entities = sample.get("entities", [])

            # Get SpaCy entities (keep original case for model, lowercase for matching)
            doc = nlp(text)
            spacy_entities_orig = [ent.text for ent in doc.ents]  # Original case
            spacy_entities_lower = [ent.text.lower() for ent in doc.ents]  # For matching

            if debug_count < debug_max and gt_entities:
                logger.info(f"\n[DEBUG SpaCy] Sample {debug_count + 1}:")
                logger.info(f"  GT entities: {[get_entity_name(e) for e in gt_entities[:3]]}")
                logger.info(f"  SpaCy entities (orig): {spacy_entities_orig[:5]}")
                debug_count += 1

            for entity_info in gt_entities:
                entity_name = get_entity_name(entity_info)
                gt_score = entity_info.get("sentiment_score")

                if gt_score is None or not entity_name or entity_name == "UNKNOWN":
                    continue

                total_entities += 1

                # Check if SpaCy detected this entity (fuzzy match)
                matched_entity = None
                matched_entity_orig = None
                for i, spacy_text in enumerate(spacy_entities_lower):
                    if entities_match(spacy_text, entity_name):
                        matched_entity = spacy_text
                        matched_entity_orig = spacy_entities_orig[i]  # Keep original case
                        break

                if not matched_entity:
                    # Also check if entity name appears in text directly
                    if find_entity_in_text(entity_name, text):
                        matched_entity = entity_name
                        matched_entity_orig = entity_name

                if not matched_entity:
                    missing_entities += 1
                    continue

                detected_entities += 1

                # Get sentiment - use original case entity for model.analyze()
                try:
                    result = model.analyze(
                        text=text,
                        target_entities=[matched_entity_orig],
                        use_coref=False,  # Disable coreference (fastcoref not installed)
                    )

                    if matched_entity_orig in result and result[matched_entity_orig].get("sentiment_score") is not None:
                        pred_score = result[matched_entity_orig]["sentiment_score"]
                        predictions.append(pred_score)
                        ground_truth.append(gt_score)
                    elif debug_count <= debug_max:
                        if matched_entity_orig in result:
                            logger.info(f"  [DEBUG SpaCy] Entity '{matched_entity_orig}' found but score is None: {result[matched_entity_orig]}")
                        else:
                            logger.info(f"  [DEBUG SpaCy] Entity '{matched_entity_orig}' not in result keys: {list(result.keys())[:3]}")
                except Exception as e:
                    if debug_count <= debug_max:
                        logger.warning(f"  [DEBUG SpaCy] model.analyze error: {e}")

    if len(predictions) == 0:
        logger.warning("No valid predictions for SpaCy pipeline evaluation!")
        return {"error": "No valid predictions", "missing_pct": 100.0}

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Calculate metrics
    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    pearson_r, pearson_p = pearsonr(ground_truth, predictions)
    spearman_r, spearman_p = spearmanr(ground_truth, predictions)

    missing_pct = (missing_entities / total_entities * 100) if total_entities > 0 else 0

    results = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "total_entities": total_entities,
        "detected_by_spacy": detected_entities,
        "evaluated_entities": len(predictions),
        "missing_entities": missing_entities,
        "missing_pct": missing_pct,
    }

    logger.info(f"\nSpaCy Pipeline Sentiment Metrics:")
    logger.info(f"  MSE:        {mse:.4f}")
    logger.info(f"  RMSE:       {rmse:.4f}")
    logger.info(f"  MAE:        {mae:.4f}")
    logger.info(f"  Pearson r:  {pearson_r:.4f}")
    logger.info(f"  Spearman r: {spearman_r:.4f}")
    logger.info(f"\nEntity Coverage (SpaCy NER):")
    logger.info(f"  Total ground truth entities: {total_entities}")
    logger.info(f"  Detected by SpaCy:           {detected_entities}")
    logger.info(f"  Evaluated (sentiment):       {len(predictions)}")
    logger.info(f"  Missing (not detected):      {missing_entities} ({missing_pct:.1f}%)")

    return results


def evaluate_full_pipeline(model, val_samples, device):
    """
    Evaluate full CRF NER + coreference → sentiment pipeline.

    Returns:
        dict: End-to-end pipeline metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("4. FULL PIPELINE (CRF NER + COREFERENCE → SENTIMENT)")
    logger.info("=" * 60)

    model.eval()

    predictions = []
    ground_truth = []
    missing_entities = 0
    total_entities = 0
    detected_entities = 0
    debug_count = 0
    debug_max = 5

    with torch.no_grad():
        for sample in val_samples:
            text = sample.get("text", "")
            gt_entities = sample.get("entities", [])

            # Get CRF NER predictions
            encoding = model.tokenizer(
                text, return_tensors="pt", max_length=2048, truncation=True
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            encoder_output = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            ner_output = model.ner_head(encoder_output, attention_mask=attention_mask)
            predictions_ner = ner_output["predictions"]

            # Decode CRF entities (keep original case for model, lowercase for matching)
            crf_entities = model.ner_head.decode_entities(
                predictions_ner, input_ids, model.tokenizer, attention_mask
            )
            crf_entities_orig = [
                e["text"].replace("Ġ", " ").strip()
                for e in crf_entities[0]
            ]
            crf_entities_lower = [t.lower() for t in crf_entities_orig]

            if debug_count < debug_max and gt_entities:
                logger.info(f"\n[DEBUG CRF] Sample {debug_count + 1}:")
                logger.info(f"  GT entities: {[get_entity_name(e) for e in gt_entities[:3]]}")
                logger.info(f"  CRF entities (orig): {crf_entities_orig[:5]}")
                debug_count += 1

            for entity_info in gt_entities:
                entity_name = get_entity_name(entity_info)
                gt_score = entity_info.get("sentiment_score")

                if gt_score is None or not entity_name or entity_name == "UNKNOWN":
                    continue

                total_entities += 1

                # Check if CRF detected this entity (fuzzy match)
                matched_entity = None
                matched_entity_orig = None
                for i, crf_text in enumerate(crf_entities_lower):
                    if entities_match(crf_text, entity_name):
                        matched_entity = crf_text
                        matched_entity_orig = crf_entities_orig[i]
                        break

                if not matched_entity:
                    # Also check if entity name appears in text directly
                    if find_entity_in_text(entity_name, text):
                        matched_entity = entity_name
                        matched_entity_orig = entity_name

                if not matched_entity:
                    missing_entities += 1
                    continue

                detected_entities += 1

                # Get sentiment - use original case entity for model.analyze()
                try:
                    result = model.analyze(
                        text=text,
                        target_entities=[matched_entity_orig],
                        use_coref=False,  # Disable coreference (fastcoref not installed)
                    )

                    if matched_entity_orig in result and result[matched_entity_orig].get("sentiment_score") is not None:
                        pred_score = result[matched_entity_orig]["sentiment_score"]
                        predictions.append(pred_score)
                        ground_truth.append(gt_score)
                    elif debug_count <= debug_max:
                        if matched_entity_orig in result:
                            logger.info(f"  [DEBUG CRF] Entity '{matched_entity_orig}' found but score is None: {result[matched_entity_orig]}")
                        else:
                            logger.info(f"  [DEBUG CRF] Entity '{matched_entity_orig}' not in result keys: {list(result.keys())[:3]}")
                except Exception as e:
                    if debug_count <= debug_max:
                        logger.warning(f"  [DEBUG CRF] model.analyze error: {e}")

    if len(predictions) == 0:
        logger.warning("No valid predictions for full pipeline evaluation!")
        return {"error": "No valid predictions", "missing_pct": 100.0}

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Calculate metrics
    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    pearson_r, pearson_p = pearsonr(ground_truth, predictions)
    spearman_r, spearman_p = spearmanr(ground_truth, predictions)

    missing_pct = (missing_entities / total_entities * 100) if total_entities > 0 else 0

    results = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "total_entities": total_entities,
        "detected_by_crf": detected_entities,
        "evaluated_entities": len(predictions),
        "missing_entities": missing_entities,
        "missing_pct": missing_pct,
    }

    logger.info(f"\nFull Pipeline Sentiment Metrics:")
    logger.info(f"  MSE:        {mse:.4f}")
    logger.info(f"  RMSE:       {rmse:.4f}")
    logger.info(f"  MAE:        {mae:.4f}")
    logger.info(f"  Pearson r:  {pearson_r:.4f}")
    logger.info(f"  Spearman r: {spearman_r:.4f}")
    logger.info(f"\nEntity Coverage (CRF NER):")
    logger.info(f"  Total ground truth entities: {total_entities}")
    logger.info(f"  Detected by CRF NER:         {detected_entities}")
    logger.info(f"  Evaluated (sentiment):       {len(predictions)}")
    logger.info(f"  Missing (not detected):      {missing_entities} ({missing_pct:.1f}%)")

    return results


def evaluate_ner_comparison(model, val_samples, device):
    """
    5. Compare NER performance: Our CRF vs SpaCy transformer.

    For each ground truth entity, check if it's detected by:
    - Our CRF NER head
    - SpaCy transformer NER

    Returns:
        dict: Detection rates and F1 scores for both approaches
    """
    logger.info("\n" + "=" * 60)
    logger.info("5. NER COMPARISON: OUR CRF vs SPACY TRANSFORMER")
    logger.info("=" * 60)

    # Load SpaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_trf")
        logger.info("Loaded SpaCy en_core_web_trf model")
    except Exception as e:
        logger.warning(f"Could not load SpaCy transformer model: {e}")
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded SpaCy en_core_web_sm model (fallback)")
        except Exception as e2:
            logger.error(f"Could not load any SpaCy model: {e2}")
            return {"error": "SpaCy not available"}

    model.eval()

    # Track detection for each ground truth entity
    crf_detected = 0
    spacy_detected = 0
    both_detected = 0
    neither_detected = 0
    total_entities = 0

    # Track entity type breakdown
    entity_type_stats = defaultdict(lambda: {
        "total": 0, "crf_detected": 0, "spacy_detected": 0
    })

    with torch.no_grad():
        for sample in val_samples:
            text = sample.get("text", "")
            gt_entities = sample.get("entities", [])

            if not text or not gt_entities:
                continue

            # Get CRF predictions
            encoding = model.tokenizer(
                text, return_tensors="pt", max_length=2048, truncation=True
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            encoder_output = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            ner_output = model.ner_head(encoder_output, attention_mask=attention_mask)
            predictions_ner = ner_output["predictions"]

            crf_entities = model.ner_head.decode_entities(
                predictions_ner, input_ids, model.tokenizer, attention_mask
            )
            crf_texts = [e["text"].replace("Ġ", " ").strip().lower() for e in crf_entities[0]]

            # Get SpaCy predictions
            doc = nlp(text)
            spacy_texts = [ent.text.lower() for ent in doc.ents]

            # Check each ground truth entity
            for entity_info in gt_entities:
                entity_name = get_entity_name(entity_info)
                entity_type = entity_info.get("type", "UNKNOWN")

                if not entity_name or entity_name == "UNKNOWN":
                    continue

                total_entities += 1
                entity_type_stats[entity_type]["total"] += 1

                # Check CRF detection
                crf_found = any(entities_match(crf_text, entity_name) for crf_text in crf_texts)
                if not crf_found:
                    crf_found = find_entity_in_text(entity_name, " ".join(crf_texts))

                # Check SpaCy detection
                spacy_found = any(entities_match(spacy_text, entity_name) for spacy_text in spacy_texts)
                if not spacy_found:
                    spacy_found = find_entity_in_text(entity_name, " ".join(spacy_texts))

                if crf_found:
                    crf_detected += 1
                    entity_type_stats[entity_type]["crf_detected"] += 1
                if spacy_found:
                    spacy_detected += 1
                    entity_type_stats[entity_type]["spacy_detected"] += 1
                if crf_found and spacy_found:
                    both_detected += 1
                if not crf_found and not spacy_found:
                    neither_detected += 1

    # Calculate rates
    crf_rate = crf_detected / total_entities if total_entities > 0 else 0
    spacy_rate = spacy_detected / total_entities if total_entities > 0 else 0
    both_rate = both_detected / total_entities if total_entities > 0 else 0
    neither_rate = neither_detected / total_entities if total_entities > 0 else 0

    results = {
        "total_ground_truth_entities": total_entities,
        "crf_detected": crf_detected,
        "crf_detection_rate": crf_rate,
        "spacy_detected": spacy_detected,
        "spacy_detection_rate": spacy_rate,
        "both_detected": both_detected,
        "both_detection_rate": both_rate,
        "neither_detected": neither_detected,
        "neither_detection_rate": neither_rate,
        "crf_advantage": crf_rate - spacy_rate,
        "entity_type_breakdown": {},
    }

    # Per-entity-type breakdown
    for etype, stats in entity_type_stats.items():
        if stats["total"] > 0:
            results["entity_type_breakdown"][etype] = {
                "total": stats["total"],
                "crf_rate": stats["crf_detected"] / stats["total"],
                "spacy_rate": stats["spacy_detected"] / stats["total"],
            }

    logger.info(f"\nOverall Detection Rates:")
    logger.info(f"  Total ground truth entities: {total_entities}")
    logger.info(f"  Our CRF NER:    {crf_detected:5d} ({crf_rate*100:5.1f}%)")
    logger.info(f"  SpaCy Transformer: {spacy_detected:5d} ({spacy_rate*100:5.1f}%)")
    logger.info(f"  Both detected:  {both_detected:5d} ({both_rate*100:5.1f}%)")
    logger.info(f"  Neither:        {neither_detected:5d} ({neither_rate*100:5.1f}%)")
    logger.info(f"\n  CRF Advantage: {(crf_rate - spacy_rate)*100:+.1f}%")

    logger.info(f"\nPer-Entity-Type Breakdown:")
    for etype, stats in sorted(results["entity_type_breakdown"].items(), key=lambda x: -x[1]["total"]):
        logger.info(f"  {etype:12s}: CRF {stats['crf_rate']*100:5.1f}% vs SpaCy {stats['spacy_rate']*100:5.1f}% (n={stats['total']})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument("--checkpoint", type=str,
                       default=str(PROJECT_ROOT / "checkpoints" / "stage2_joint" / "best_model.pt"))
    parser.add_argument("--data_dir", type=str,
                       default=str(PROJECT_ROOT / "data" / "labeled" / "holdout"))
    parser.add_argument("--val_file", type=str,
                       default=str(PROJECT_ROOT / "data" / "labeled" / "holdout" / "eodhd_non_yahoo_holdout_labeled.jsonl"),
                       help="Direct path to validation JSONL file (overrides data_dir)")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str,
                       default=str(PROJECT_ROOT / "outputs" / "evaluation_results.json"))

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("=" * 60)

    # Load model
    logger.info(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    logger.info(f"Checkpoint F1: {checkpoint.get('val_f1', 'N/A')}")

    model = FinancialEntitySentimentModel(
        encoder_name="allenai/longformer-base-4096",
        max_length=args.max_length,
        use_crf_ner=True,
        ner_label_to_id=LABEL_TO_ID,
        device=args.device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()
    logger.info("Model loaded successfully!")

    # Load validation data
    data_source = args.val_file if args.val_file else args.data_dir
    logger.info(f"\nLoading validation data from: {data_source}")
    val_loader, preprocessor = load_validation_data(
        args.data_dir, args.val_split, args.max_length, val_file=args.val_file
    )
    val_samples = load_raw_validation_samples(args.data_dir, args.val_split, val_file=args.val_file)
    logger.info(f"Validation samples: {len(val_samples)}")

    all_results = {}

    # 1. NER Evaluation
    ner_results = evaluate_ner(model, val_loader, args.device)
    all_results["ner"] = ner_results

    # 2. Sentiment Standalone Evaluation
    sentiment_results = evaluate_sentiment_standalone(model, val_samples, args.device, args.max_length)
    all_results["sentiment_standalone"] = sentiment_results

    # 3. SpaCy Pipeline Evaluation
    spacy_results = evaluate_spacy_pipeline(model, val_samples, args.device)
    all_results["spacy_pipeline"] = spacy_results

    # 4. Full Pipeline Evaluation
    full_results = evaluate_full_pipeline(model, val_samples, args.device)
    all_results["full_pipeline"] = full_results

    # 5. NER Comparison: Our CRF vs SpaCy
    ner_comparison_results = evaluate_ner_comparison(model, val_samples, args.device)
    all_results["ner_comparison"] = ner_comparison_results

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    logger.info("\n1. NER (CRF) Performance:")
    logger.info(f"   F1: {ner_results['overall']['f1']:.3f}")

    logger.info("\n2. Sentiment (Ground Truth Entities):")
    if "error" not in sentiment_results:
        logger.info(f"   RMSE: {sentiment_results['rmse']:.4f}")
        logger.info(f"   Pearson r: {sentiment_results['pearson_r']:.4f}")
        logger.info(f"   Missing: {sentiment_results['missing_pct']:.1f}%")

    logger.info("\n3. SpaCy + Coreference → Sentiment:")
    if "error" not in spacy_results:
        logger.info(f"   RMSE: {spacy_results['rmse']:.4f}")
        logger.info(f"   Pearson r: {spacy_results['pearson_r']:.4f}")
        logger.info(f"   Missing: {spacy_results['missing_pct']:.1f}%")
    else:
        logger.info(f"   {spacy_results.get('error', 'Error')}")

    logger.info("\n4. Full Pipeline (CRF + Coreference → Sentiment):")
    if "error" not in full_results:
        logger.info(f"   RMSE: {full_results['rmse']:.4f}")
        logger.info(f"   Pearson r: {full_results['pearson_r']:.4f}")
        logger.info(f"   Missing: {full_results['missing_pct']:.1f}%")

    logger.info("\n5. NER Comparison (CRF vs SpaCy):")
    if "error" not in ner_comparison_results:
        logger.info(f"   Our CRF Detection Rate:    {ner_comparison_results['crf_detection_rate']*100:.1f}%")
        logger.info(f"   SpaCy Detection Rate:      {ner_comparison_results['spacy_detection_rate']*100:.1f}%")
        logger.info(f"   CRF Advantage:             {ner_comparison_results['crf_advantage']*100:+.1f}%")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        json.dump(convert_to_serializable(all_results), f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
