#!/usr/bin/env python3
"""Validate source classification rules against Haiku-labeled ground truth.

Applies a set of regex rules to a validation sample and computes precision,
recall, and F1 per rule and overall.

Usage:
    python scripts/analysis/validate_source_rules.py \
        --validation-file outputs/validation_sample_haiku.jsonl \
        --rules-file outputs/candidate_source_rules.json \
        --output outputs/rule_validation_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("validate_rules")


@dataclass
class ValidationRule:
    """A rule loaded from JSON for validation."""
    name: str
    pattern: str
    target_content_type: str
    target_tier: int
    _compiled: Any = None

    def __post_init__(self):
        flags = re.IGNORECASE
        # If pattern starts with ^, search in content directly
        self._compiled = re.compile(self.pattern, flags)

    def match(self, title: str, content: str, url: str = "", url_domain: str = "") -> bool:
        if self.pattern.startswith("^"):
            text = content
        else:
            body = f"{title}\n{content}" if title or content else ""
            body = body[:1000]
            # Include URL fields so patterns like 'benzinga.com', 'fool.com', or
            # 'seekingalpha.com/article' can match against the article URL even
            # when the body itself does not carry the source attribution.
            text = f"{url}\n{url_domain}\n{body}"
        return bool(self._compiled.search(text))


def load_rules(rules_file: Path) -> List[ValidationRule]:
    """Load rules from a JSON file."""
    with open(rules_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support two formats: flat list of rules or aggregated candidate output
    rules_data = data if isinstance(data, list) else data.get("rules", [])

    rules: List[ValidationRule] = []
    for item in rules_data:
        rules.append(ValidationRule(
            name=item["name"],
            pattern=item["pattern"],
            target_content_type=item.get("target_content_type", item.get("content_type", "other")),
            target_tier=int(item.get("target_tier", item.get("tier", 3))),
        ))
    return rules


def load_validation_records(val_file: Path) -> List[Dict[str, Any]]:
    """Load validation records with Haiku ground truth."""
    records: List[Dict[str, Any]] = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("haiku_classification"):
                    records.append(rec)
    return records


def evaluate_rules(
    rules: List[ValidationRule],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate rules against ground truth."""
    # Per-rule stats
    rule_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    for rec in records:
        gt = rec["haiku_classification"].get("content_type", "other")
        title = rec.get("title", "")
        content = rec.get("content", "")

        url = rec.get("url", "")
        url_domain = rec.get("url_domain", "")

        matched = False
        for rule in rules:
            if rule.match(title, content, url, url_domain):
                matched = True
                if rule.target_content_type == gt:
                    rule_stats[rule.name]["tp"] += 1
                    overall_tp += 1
                else:
                    rule_stats[rule.name]["fp"] += 1
                    overall_fp += 1
                break  # first match wins

        if not matched:
            # Count as FN for any rule that should have matched this category
            for rule in rules:
                if rule.target_content_type == gt:
                    rule_stats[rule.name]["fn"] += 1
                    overall_fn += 1

    # Compute metrics
    def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    per_rule = {}
    for rule in rules:
        stats = rule_stats[rule.name]
        per_rule[rule.name] = {
            "target": rule.target_content_type,
            **compute_metrics(stats["tp"], stats["fp"], stats["fn"]),
        }

    overall = compute_metrics(overall_tp, overall_fp, overall_fn)

    # Per-category stats (regardless of which rule matched)
    category_preds: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for rec in records:
        gt = rec["haiku_classification"].get("content_type", "other")
        title = rec.get("title", "")
        content = rec.get("content", "")

        url = rec.get("url", "")
        url_domain = rec.get("url_domain", "")

        pred = "other"
        for rule in rules:
            if rule.match(title, content, url, url_domain):
                pred = rule.target_content_type
                break

        for cat in set([gt, pred]):
            if pred == cat and gt == cat:
                category_preds[cat]["tp"] += 1
            elif pred == cat and gt != cat:
                category_preds[cat]["fp"] += 1
            elif pred != cat and gt == cat:
                category_preds[cat]["fn"] += 1

    per_category = {}
    for cat, stats in category_preds.items():
        per_category[cat] = compute_metrics(stats["tp"], stats["fp"], stats["fn"])

    # Tier-level confusion: what tier does each rule predict vs the tier
    # implied by the ground-truth content_type? This is the metric that
    # downstream model weighting actually uses.
    ct_to_tier = {r.target_content_type: r.target_tier for r in rules}
    # Manual tier assignment for ground-truth categories not covered by rules.
    gt_tier_map = dict(ct_to_tier)
    gt_tier_map.update({
        "yahoo_market_wrap": 2, "yahoo_stock_bulletin": 3,
        "yahoo_analyst_call": 2, "other": 3,
    })

    tier_confusion: Dict[str, int] = defaultdict(int)  # "gt->pred"
    tier_correct = 0
    tier_total = 0
    for rec in records:
        gt = rec["haiku_classification"].get("content_type", "other")
        gt_tier = gt_tier_map.get(gt, 3)
        title = rec.get("title", "")
        content = rec.get("content", "")
        url = rec.get("url", "")
        url_domain = rec.get("url_domain", "")
        pred_tier = None
        for rule in rules:
            if rule.match(title, content, url, url_domain):
                pred_tier = rule.target_tier
                break
        if pred_tier is None:
            # No rule matched. Default to T3 (low-quality) for unclassified.
            pred_tier = 3
        tier_confusion[f"{gt_tier}->{pred_tier}"] += 1
        tier_total += 1
        if gt_tier == pred_tier:
            tier_correct += 1

    tier_accuracy = tier_correct / tier_total if tier_total else 0.0

    return {
        "overall": overall,
        "per_rule": per_rule,
        "per_category": per_category,
        "tier_accuracy": tier_accuracy,
        "tier_confusion": dict(tier_confusion),
        "total_evaluated": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate source classification rules")
    parser.add_argument(
        "--validation-file",
        type=Path,
        required=True,
        help="JSONL with Haiku ground-truth classifications",
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        required=True,
        help="JSON file with rules to validate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON report",
    )
    args = parser.parse_args()

    rules = load_rules(args.rules_file)
    records = load_validation_records(args.validation_file)
    logger.info(f"Loaded {len(rules)} rules and {len(records)} validation records")

    report = evaluate_rules(rules, records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Wrote report to {args.output}")
    logger.info(f"Overall precision: {report['overall']['precision']:.2%}")
    logger.info(f"Overall recall:    {report['overall']['recall']:.2%}")
    logger.info(f"Overall F1:        {report['overall']['f1']:.2%}")
    logger.info(f"Tier-level accuracy: {report['tier_accuracy']:.2%}")
    logger.info(f"Tier confusion (gt->pred): {report['tier_confusion']}")

    logger.info("Per-rule precision:")
    for name, metrics in sorted(report["per_rule"].items(), key=lambda x: -x[1]["precision"]):
        logger.info(
            f"  {name:30s} | P={metrics['precision']:.2%} R={metrics['recall']:.2%} "
            f"F1={metrics['f1']:.2%} (tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']})"
        )


if __name__ == "__main__":
    main()
