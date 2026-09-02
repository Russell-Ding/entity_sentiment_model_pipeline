#!/usr/bin/env python3
"""Apply Phase-2 regex rules to rewrite source tiers in collected articles.

Reads each <TICKER>.jsonl.gz, applies validated classification rules,
and writes updated records with `final_source_tier`, `rule_name`,
`content_type`, and `syndication_source`.

Usage:
    python scripts/preprocessing/apply_source_tiers.py \
        --input-dir data/raw/eodhd_bulk_20260517/news \
        --rules-file outputs/source_rules.json \
        --output-dir data/raw/eodhd_bulk_20260517/news_retiered
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("apply_tiers")


@dataclass
class SourceRule:
    """A single source classification rule."""
    name: str
    pattern: str
    tier: int
    detected_source: str
    content_type: str
    syndication_source: Optional[str] = None

    def __post_init__(self):
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def match(self, title: str, content: str, url_domain: str, url: str = "") -> bool:
        # Prefix rules (anchored with ^) search content directly so the title
        # does not shift the anchor position.
        if self.pattern.startswith("^"):
            text = content
        else:
            body = f"{title}\n{content}" if title or content else ""
            body = body[:1000]
            # Prepend URL fields so domain patterns (benzinga.com, fool.com,
            # seekingalpha.com/article) match against the URL when the body
            # itself does not contain explicit attribution.
            text = f"{url}\n{url_domain}\n{body}"
        return bool(self._compiled.search(text))


DEFAULT_RULES: List[SourceRule] = [
    # Wire prefixes (highest priority)
    SourceRule("bloomberg_prefix", r"^\s*\(Bloomberg(?:\s+Opinion)?\)", 1, "Bloomberg", "bloomberg", "Bloomberg"),
    SourceRule("reuters_prefix", r"^\s*\(Reuters\)", 1, "Reuters", "reuters", "Reuters"),
    SourceRule("wsj_prefix", r"^\s*\(WSJ\)", 1, "WSJ", "wsj", "WSJ"),
    SourceRule("ft_prefix", r"^\s*\(FT\)", 1, "FT", "ft", "FT"),
    SourceRule("ap_prefix", r"^\s*\(AP\)", 1, "AP", "ap", "AP"),
    SourceRule("afp_prefix", r"^\s*\(AFP\)", 1, "AFP", "afp", "AFP"),
    # Body attributions
    SourceRule("marketwatch_body", r"MarketWatch", 1, "MarketWatch", "marketwatch"),
    # Zacks Investment Research → Tier 3, not Tier 2. Empirically these are
    # algo-generated stock-movement bulletins ("X closed at $Y, +Z%") and SEO
    # templates ("3 dividend stocks..."), not analyst-grade content. Same as
    # the constants in collect_eodhd_sp500_bulk.py.
    SourceRule("zacks_investment_research", r"Zacks\s+Investment\s+Research", 3, "Zacks Investment Research", "zacks_press_release"),
    SourceRule("zacks_equity_research", r"Zacks\s+Equity\s+Research", 3, "Zacks Equity Research", "zacks_press_release"),
    SourceRule("benzinga_body", r"Benzinga", 2, "Benzinga", "benzinga"),
    SourceRule("globenewswire_body", r"GlobeNewswire", 2, "GlobeNewswire", "press_release_other"),
    SourceRule("prnewswire_body", r"PR\s*Newswire", 3, "PR Newswire", "press_release_other"),
    SourceRule("businesswire_body", r"Business\s+Wire", 3, "Business Wire", "press_release_other"),
    SourceRule("motleyfool_body", r"fool\.com|Motley\s+Fool", 2, "Motley Fool", "motleyfool"),
]


def load_rules(rules_file: Optional[Path]) -> List[SourceRule]:
    """Load rules from JSON file or return defaults."""
    if not rules_file or not rules_file.exists():
        logger.info("Using default source rules")
        return DEFAULT_RULES

    with open(rules_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rules: List[SourceRule] = []
    for item in data:
        # Accept both validator-style (target_tier/target_content_type) and
        # applier-style (tier/content_type) field names so the same JSON
        # works in both scripts.
        rules.append(SourceRule(
            name=item["name"],
            pattern=item["pattern"],
            tier=int(item.get("tier", item.get("target_tier", 3))),
            detected_source=item.get("detected_source", item.get("name", "unknown")),
            content_type=item.get("content_type", item.get("target_content_type", "other")),
            syndication_source=item.get("syndication_source"),
        ))
    logger.info(f"Loaded {len(rules)} rules from {rules_file}")
    return rules


def apply_rules(
    record: Dict[str, Any],
    rules: List[SourceRule],
) -> Dict[str, Any]:
    """Apply rules to a single record. Returns updated fields dict."""
    title = record.get("title", "")
    content = record.get("content", "")
    url_domain = record.get("url_domain", "")
    url = record.get("url", "")

    for rule in rules:
        if rule.match(title, content, url_domain, url):
            return {
                "rule_name": rule.name,
                "final_source_tier": rule.tier,
                "detected_source": rule.detected_source,
                "content_type": rule.content_type,
                "syndication_source": rule.syndication_source,
            }

    # No rule matched: keep phase-1 values
    return {
        "rule_name": None,
        "final_source_tier": record.get("source_tier", 3),
        "detected_source": record.get("detected_source", "unknown"),
        "content_type": "other",
        "syndication_source": None,
    }


def process_file(
    input_path: Path,
    output_path: Path,
    rules: List[SourceRule],
) -> Dict[str, int]:
    """Process one ticker file. Returns counter of rule matches."""
    counter: Dict[str, int] = defaultdict(int)
    tmp_path = output_path.with_suffix(".tmp")

    with gzip.open(input_path, "rt", encoding="utf-8") as f_in:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                rec = json.loads(line)
                updates = apply_rules(rec, rules)
                rec.update(updates)
                counter[updates["rule_name"] or "no_rule"] += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    os.replace(str(tmp_path), str(output_path))
    return dict(counter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply source tier rules to collected articles")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing original <TICKER>.jsonl.gz files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write retiered <TICKER>.jsonl.gz files",
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        default=None,
        help="JSON file with custom rules (default: use built-in rules)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show rule summary without rewriting files",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    rules = load_rules(args.rules_file)

    if args.dry_run:
        logger.info("Rules that will be applied:")
        for r in rules:
            logger.info(f"  {r.name}: tier={r.tier} pattern={r.pattern}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_records = 0
    global_counter: Dict[str, int] = defaultdict(int)

    input_files = sorted(args.input_dir.glob("*.jsonl.gz"))
    logger.info(f"Processing {len(input_files)} files...")

    for input_path in input_files:
        output_path = args.output_dir / input_path.name
        counter = process_file(input_path, output_path, rules)
        for k, v in counter.items():
            global_counter[k] += v
        total_records += sum(counter.values())
        logger.info(f"  {input_path.name}: {sum(counter.values())} records")

    logger.info(f"Done. Total records processed: {total_records}")
    logger.info("Rule match distribution:")
    for rule_name, count in sorted(global_counter.items(), key=lambda x: -x[1]):
        pct = count / total_records * 100 if total_records else 0
        logger.info(f"  {rule_name}: {count:,} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
