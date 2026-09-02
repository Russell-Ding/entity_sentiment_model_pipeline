#!/usr/bin/env python3
"""Classify article sources using Claude Haiku 4.5.

Reads a sampled JSONL file, sends each article to Haiku for source
classification, and writes enriched results.

Usage:
    export ANTHROPIC_API_KEY="..."
    python scripts/labeling/classify_sources_haiku.py \
        --input outputs/source_classification_sample.jsonl \
        --output outputs/source_classification_haiku.jsonl \
        --model claude-3-5-haiku-latest
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("classify_sources")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import anthropic
except ImportError:
    logger.error("anthropic not installed. Run: pip install anthropic")
    sys.exit(1)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


CLASSIFICATION_PROMPT = """You are a financial news source classifier.

Analyze the following article excerpt and classify its original source / content type.

Return ONLY a JSON object with these exact fields:
{{
  "content_type": string,
  "confidence": "high" | "medium" | "low",
  "other_description": string or null
}}

content_type must be ONE of:
- bloomberg — Bloomberg wire content (starts with "(Bloomberg)" or "(Bloomberg Opinion)" — same applies to articles rehosted via Yahoo Finance)
- reuters — Reuters wire content (starts with "(Reuters)" — same applies to Yahoo-rehosted Reuters)
- ap — Associated Press wire content
- wsj — Wall Street Journal content
- ft — Financial Times content
- barrons — Barron's premium financial publication (Dow Jones)
- marketwatch — MarketWatch original content
- mt_newswires — MT Newswires premium financial wire content (often subscription-gated, attribution to "MT Newswires")
- motleyfool — Motley Fool / fool.com content
- seeking_alpha — Seeking Alpha user-contributed analyst articles (opinion/contrarian takes, "SA Premium")
- ibd — Investor's Business Daily content (CAN SLIM analysis, "Dow Jones Futures" series, buy points, chart patterns)
- investing_com — Investing.com original content (market summaries, technical analysis)
- benzinga — Benzinga news and bulletins (attribution to Benzinga, fast-moving market briefs)
- barchart — Barchart.com algorithmic market summaries (embedded ticker links, index performance tables)
- earnings_transcript — Verbatim earnings call transcript (Q&A format, "Operator:", "CEO:" speaker tags)
- gurufocus — GuruFocus stock analysis (GF Score metric, guru-tracking, factor screens)
- insider_monkey — Insider Monkey 13F hedge fund analysis (hedge fund holdings, fund manager picks)
- simplywallst — Simply Wall St algorithmic stock reports (DCF/Snowflake visualizations, intrinsic value)
- zacks_press_release — Zacks syndicated press release (contains "Zacks Equity Research")
- yahoo_market_wrap — Yahoo Finance broad market summary (mentions multiple tickers, no clear authorship)
- yahoo_stock_bulletin — Yahoo algo-generated stock movement bulletin (price data, no analysis)
- yahoo_analyst_call — Yahoo summary of an analyst call or note (cites Goldman, JPM, etc.)
- yahoo_seo_promotional — Yahoo SEO listicle / promotional content ("3 stocks to buy", "top dividend picks")
- press_release_other — Other press release (GlobeNewswire, PR Newswire, Business Wire, "For Immediate Release")
- other — None of the above

For wire-syndicated content (Bloomberg, Reuters, AP, WSJ, FT, MarketWatch),
ALWAYS classify by the underlying wire service, NOT as a "yahoo_*" category,
even when the article is rehosted via Yahoo Finance.

confidence: your certainty in this classification.

other_description: if content_type is "other", briefly describe what it is (1 sentence). Otherwise null.

Hints:
- Press releases often contain "For Immediate Release", "GlobeNewswire", "PR Newswire", or "Zacks Equity Research" in the first 200 characters.
- Bloomberg wire stories start with "(Bloomberg)" or "(Bloomberg Opinion)".
- Yahoo market wraps have no clear byline and summarize multiple companies.
- Yahoo stock bulletins quote specific prices and percentages with minimal narrative.
- SEO content uses listicle titles ("3 great dividend stocks", "5 stocks to buy now").

Title:
{title}

Content (first 500 characters):
{content}

JSON:"""


def get_api_key(provider: str = "kimi") -> Optional[str]:
    """Get API key for the chosen provider.

    Priority: env var (KIMI_API_KEY / ANTHROPIC_API_KEY) → config/secrets.yaml.
    """
    env_key = "KIMI_API_KEY" if provider == "kimi" else "ANTHROPIC_API_KEY"
    yaml_key = "kimi_api_key" if provider == "kimi" else "anthropic_key"

    api_key = os.environ.get(env_key)
    if api_key:
        return api_key

    secrets_path = PROJECT_ROOT / "config" / "secrets.yaml"
    if secrets_path.exists() and HAS_YAML:
        try:
            with open(secrets_path) as f:
                secrets = yaml.safe_load(f)
                return secrets.get("api", {}).get(yaml_key)
        except Exception:
            pass
    return None


# Provider → (default model, Anthropic-SDK base_url). Kimi For Coding accepts
# Anthropic-protocol /v1/messages requests at api.kimi.com/coding, so we just
# point the Anthropic SDK there and Moonshot serves Kimi K2.
PROVIDER_CONFIG = {
    "kimi":      ("kimi-for-coding",         "https://api.kimi.com/coding"),
    "anthropic": ("claude-haiku-4-5",        None),  # default Anthropic base
}


class SourceClassifier:
    """Classifies article sources via Anthropic SDK (works for Kimi too)."""

    def __init__(
        self,
        api_key: str,
        provider: str = "kimi",
        model: Optional[str] = None,
        max_retries: int = 3,
        delay: float = 0.5,
    ):
        default_model, base_url = PROVIDER_CONFIG[provider]
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model or default_model
        self.provider = provider
        self.max_retries = max_retries
        self.delay = delay

    def classify(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Classify a single article. Returns classification dict or None."""
        title = article.get("title", "")
        content = article.get("content", "")

        if not title and not content:
            return None

        prompt = CLASSIFICATION_PROMPT.format(
            title=title,
            content=content[:500] if content else "",
        )

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.delay)

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )

                response_text = response.content[0].text.strip()

                # Strip markdown code fences if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                result = json.loads(response_text)

                # Validate expected keys
                required = {"content_type", "confidence", "other_description"}
                if not required.issubset(result.keys()):
                    logger.warning(f"Missing keys in response (attempt {attempt + 1})")
                    continue

                # Normalize content_type
                allowed = {
                    "bloomberg", "reuters", "ap", "wsj", "ft", "barrons",
                    "marketwatch", "mt_newswires",
                    "motleyfool", "seeking_alpha", "ibd", "investing_com",
                    "benzinga", "barchart", "earnings_transcript",
                    "gurufocus", "insider_monkey", "simplywallst",
                    "zacks_press_release", "yahoo_market_wrap",
                    "yahoo_stock_bulletin", "yahoo_analyst_call",
                    "yahoo_seo_promotional", "press_release_other", "other",
                }
                ct = result.get("content_type", "other")
                if ct not in allowed:
                    logger.warning(f"Unexpected content_type '{ct}', defaulting to 'other'")
                    result["content_type"] = "other"

                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                # Brief jittered pause — model produced bad JSON, retry quickly
                time.sleep(random.uniform(0.2, 0.8))
                continue
            except anthropic.APIError as e:
                # Exponential backoff with jitter: 2s, 4s, 8s, capped at 60s
                base = min(2 ** (attempt + 1), 60)
                wait = base + random.uniform(0, base * 0.25)
                is_rate_limit = "rate_limit" in str(e).lower() or getattr(e, "status_code", None) == 429
                level = "warning" if is_rate_limit else "warning"
                logger.log(
                    logging.WARNING,
                    f"API error (attempt {attempt + 1}, sleeping {wait:.1f}s): {e}",
                )
                time.sleep(wait)
                continue
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt + 1}): {e}")
                time.sleep(random.uniform(0.5, 1.5))
                continue

        return None


def process_batch(
    classifier: SourceClassifier,
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Process a batch of articles sequentially."""
    results: List[Dict[str, Any]] = []
    for article in articles:
        classification = classifier.classify(article)
        if classification:
            article["haiku_classification"] = classification
            article["classified_at"] = datetime.now().astimezone().isoformat()
            article["classifier_model"] = classifier.model
        else:
            article["haiku_classification"] = None
            article["classification_error"] = True
        results.append(article)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify article sources with Haiku")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with sampled articles",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file with classifications",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="kimi",
        choices=["kimi", "anthropic"],
        help="LLM provider (default: kimi → uses kimi_api_key from secrets.yaml, "
             "calls api.kimi.com/coding with kimi-for-coding model). Use "
             "'anthropic' for Claude Haiku via api.anthropic.com.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name. Defaults: kimi-for-coding (kimi) or "
             "claude-haiku-4-5 (anthropic).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Articles per worker batch (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-classified articles in output file",
    )
    args = parser.parse_args()

    api_key = get_api_key(provider=args.provider)
    if not api_key:
        env_name = "KIMI_API_KEY" if args.provider == "kimi" else "ANTHROPIC_API_KEY"
        yaml_name = "kimi_api_key" if args.provider == "kimi" else "anthropic_key"
        raise SystemExit(
            f"{env_name} not found. Set env var or add api.{yaml_name} to config/secrets.yaml"
        )

    # Load articles
    articles: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    logger.info(f"Loaded {len(articles)} articles from {args.input}")

    # Resume support
    done_ids: set = set()
    if args.resume and args.output.exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("haiku_classification"):
                        done_ids.add(rec.get("id"))
        logger.info(f"Resuming: {len(done_ids)} already classified")

    todo = [a for a in articles if a.get("id") not in done_ids]
    logger.info(f"Articles to classify: {len(todo)}")

    if not todo:
        logger.info("Nothing to do.")
        return

    classifier = SourceClassifier(
        api_key=api_key, provider=args.provider, model=args.model,
    )

    # Split into batches
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(todo), args.batch_size):
        batches.append(todo[i : i + args.batch_size])

    # Process batches with thread pool
    results: List[Dict[str, Any]] = []
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_batch, classifier, batch): idx
            for idx, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            batch_idx = futures[fut]
            try:
                batch_results = fut.result()
                results.extend(batch_results)
                completed += sum(1 for r in batch_results if r.get("haiku_classification"))
                failed += sum(1 for r in batch_results if not r.get("haiku_classification"))
                logger.info(f"Batch {batch_idx + 1}/{len(batches)} done | success={completed} failed={failed}")
            except Exception as e:
                logger.exception(f"Batch {batch_idx + 1} failed: {e}")

    # Combine with previously completed if resuming
    if args.resume and done_ids:
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id") in done_ids:
                        results.append(rec)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(results)} records to {args.output}")
    logger.info(f"Classified: {completed} | Failed: {failed}")


if __name__ == "__main__":
    main()
