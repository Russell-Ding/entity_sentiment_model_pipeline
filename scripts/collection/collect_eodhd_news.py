#!/usr/bin/env python3
"""EODHD News Collector - Optimized for Free Tier (4 requests/day).

Maximizes data collection with strategic query planning:
- 1,000 articles per request (max allowed)
- 4 requests/day = 4,000 articles potential
- Multi-day collection tracking
- Date range validation
- Deduplication across sessions

Usage:
    python collect_eodhd_news.py                    # Run with default strategy
    python collect_eodhd_news.py --strategy topics  # Use topic-based collection
    python collect_eodhd_news.py --strategy tickers # Use ticker-based collection
    python collect_eodhd_news.py --dry-run          # Show plan without executing

API Key: Set in config/secrets.yaml as api.eodhd_api
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
except ImportError:
    logger.error("requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Collection strategies optimized for 4 requests/day
STRATEGIES = {
    "topics": {
        "description": "Collect by financial topics (broad coverage)",
        "queries": [
            {"t": "earnings", "description": "Earnings announcements"},
            {"t": "mergers and acquisitions", "description": "M&A news"},
            {"t": "technology", "description": "Tech sector news"},
            {"t": "finance", "description": "Finance sector news"},
        ]
    },
    "tickers": {
        "description": "Collect by major tickers (company-specific)",
        "queries": [
            {"s": "AAPL.US", "description": "Apple Inc."},
            {"s": "NVDA.US", "description": "NVIDIA Corporation"},
            {"s": "MSFT.US", "description": "Microsoft Corporation"},
            {"s": "GOOGL.US", "description": "Alphabet Inc."},
        ]
    },
    "historical": {
        "description": "Collect by time periods (deep history)",
        "queries": [
            {"from": "2020-10-01", "to": "2021-09-30", "description": "Oct 2020 - Sep 2021"},
            {"from": "2021-10-01", "to": "2022-12-31", "description": "Oct 2021 - Dec 2022"},
            {"from": "2023-01-01", "to": "2024-06-30", "description": "Jan 2023 - Jun 2024"},
            {"from": "2024-07-01", "to": "2026-01-31", "description": "Jul 2024 - Jan 2026"},
        ]
    },
    "hybrid": {
        "description": "Mixed strategy for diversity (recommended)",
        "queries": [
            {"t": "earnings", "description": "Earnings news (all time)"},
            {"s": "NVDA.US", "description": "NVIDIA (AI leader)"},
            {"s": "JPM.US", "description": "JPMorgan (Finance leader)"},
            {"t": "mergers and acquisitions", "description": "M&A news (all time)"},
        ]
    },
    "sector_rotation": {
        "description": "Rotate through sectors daily",
        "queries": [
            {"t": "technology", "description": "Technology sector"},
            {"t": "healthcare", "description": "Healthcare sector"},
            {"t": "energy", "description": "Energy sector"},
            {"t": "finance", "description": "Finance sector"},
        ]
    }
}

# High-value tickers for rotation
MAJOR_TICKERS = [
    "AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US", "NVDA.US",
    "META.US", "TSLA.US", "JPM.US", "V.US", "JNJ.US",
    "UNH.US", "HD.US", "PG.US", "MA.US", "BAC.US",
    "XOM.US", "CVX.US", "PFE.US", "ABBV.US", "KO.US",
    "DIS.US", "NFLX.US", "AMD.US", "INTC.US", "CRM.US",
    "GS.US", "MS.US", "WFC.US", "C.US", "BLK.US",
]


def get_api_key() -> Optional[str]:
    """Get EODHD API key from environment or secrets file."""
    api_key = os.environ.get("EODHD_API_KEY")
    if api_key:
        return api_key

    secrets_path = PROJECT_ROOT / "config" / "secrets.yaml"
    if secrets_path.exists() and HAS_YAML:
        try:
            with open(secrets_path) as f:
                secrets = yaml.safe_load(f)
                return secrets.get("api", {}).get("eodhd_api")
        except Exception:
            pass
    return None


def get_api_usage(api_key: str) -> Dict:
    """Get current API usage statistics."""
    url = f"https://eodhd.com/api/user?api_token={api_key}&fmt=json"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get API usage: {e}")
        return {}


def get_remaining_requests(api_key: str) -> int:
    """Get remaining news API requests for today."""
    usage = get_api_usage(api_key)
    daily_limit = usage.get("dailyRateLimit", 20)
    used = usage.get("apiRequests", 0)
    remaining_calls = daily_limit - used
    return remaining_calls // 5  # Each news request costs 5 API calls


def load_collection_state() -> Dict:
    """Load collection state to track multi-day progress."""
    state_file = PROJECT_ROOT / "data" / "eodhd_collection_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "collected_urls": [],
        "last_collection": None,
        "total_articles": 0,
        "queries_completed": [],
    }


def save_collection_state(state: Dict) -> None:
    """Save collection state for multi-day tracking."""
    state_file = PROJECT_ROOT / "data" / "eodhd_collection_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def fetch_news(
    api_key: str,
    ticker: Optional[str] = None,
    topic: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
) -> Tuple[List[Dict], bool]:
    """
    Fetch news from EODHD API.

    Returns:
        Tuple of (articles_list, success_bool)
    """
    url = "https://eodhd.com/api/news"
    params = {
        "api_token": api_key,
        "limit": min(limit, 1000),  # Max 1000 per request
        "offset": offset,
        "fmt": "json",
    }

    if ticker:
        params["s"] = ticker
    if topic:
        params["t"] = topic
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            return data, True
        else:
            logger.warning(f"Unexpected response format: {type(data)}")
            return [], False

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        return [], False
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return [], False


def validate_date_range(articles: List[Dict], expected_from: str = None, expected_to: str = None) -> Dict:
    """Validate the date range of collected articles."""
    if not articles:
        return {"valid": False, "reason": "No articles"}

    dates = []
    for a in articles:
        date_str = a.get("date", "")[:10]
        if date_str:
            dates.append(date_str)

    if not dates:
        return {"valid": False, "reason": "No dates found"}

    actual_min = min(dates)
    actual_max = max(dates)

    result = {
        "valid": True,
        "actual_from": actual_min,
        "actual_to": actual_max,
        "article_count": len(articles),
        "unique_dates": len(set(dates)),
    }

    if expected_from and actual_min < expected_from:
        result["warning"] = f"Articles older than expected: {actual_min} < {expected_from}"
    if expected_to and actual_max > expected_to:
        result["warning"] = f"Articles newer than expected: {actual_max} > {expected_to}"

    return result


def collect_with_strategy(
    api_key: str,
    strategy_name: str = "hybrid",
    max_requests: Optional[int] = None,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Execute collection with specified strategy.

    Args:
        api_key: EODHD API key
        strategy_name: Name of strategy from STRATEGIES dict
        max_requests: Override max requests (default: use all remaining)
        dry_run: If True, show plan without executing

    Returns:
        List of collected articles
    """
    if strategy_name not in STRATEGIES:
        logger.error(f"Unknown strategy: {strategy_name}")
        logger.info(f"Available strategies: {list(STRATEGIES.keys())}")
        return []

    strategy = STRATEGIES[strategy_name]
    queries = strategy["queries"]

    # Check remaining requests
    remaining = get_remaining_requests(api_key)
    if max_requests:
        remaining = min(remaining, max_requests)

    logger.info("=" * 60)
    logger.info(f"EODHD NEWS COLLECTION - Strategy: {strategy_name}")
    logger.info("=" * 60)
    logger.info(f"Description: {strategy['description']}")
    logger.info(f"Remaining requests today: {remaining}")
    logger.info(f"Queries planned: {len(queries)}")
    logger.info(f"Max potential articles: {remaining * 1000}")

    if remaining <= 0:
        logger.warning("No API requests remaining today!")
        logger.info("API resets at midnight UTC. Try again tomorrow.")
        return []

    # Show plan
    logger.info("\nCollection Plan:")
    for i, q in enumerate(queries[:remaining]):
        desc = q.get("description", "")
        params = {k: v for k, v in q.items() if k != "description"}
        logger.info(f"  Request {i+1}: {desc}")
        logger.info(f"           Params: {params}")

    if dry_run:
        logger.info("\n[DRY RUN] No requests executed.")
        return []

    # Load state for deduplication
    state = load_collection_state()
    seen_urls = set(state.get("collected_urls", []))

    # Execute collection
    all_articles = []
    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eodhd_news_{timestamp}.jsonl"

    for i, query in enumerate(queries[:remaining]):
        logger.info(f"\n--- Request {i+1}/{min(len(queries), remaining)} ---")
        desc = query.pop("description", "")
        logger.info(f"Fetching: {desc}")

        # Build query params
        ticker = query.get("s")
        topic = query.get("t")
        from_date = query.get("from")
        to_date = query.get("to")

        # Fetch with max limit
        articles, success = fetch_news(
            api_key=api_key,
            ticker=ticker,
            topic=topic,
            from_date=from_date,
            to_date=to_date,
            limit=1000,
        )

        if not success:
            logger.warning(f"Request failed for: {desc}")
            continue

        # Validate date range
        validation = validate_date_range(articles, from_date, to_date)
        logger.info(f"  Received: {len(articles)} articles")
        logger.info(f"  Date range: {validation.get('actual_from', 'N/A')} to {validation.get('actual_to', 'N/A')}")
        if validation.get("warning"):
            logger.warning(f"  {validation['warning']}")

        # Deduplicate and format
        new_count = 0
        for article in articles:
            url = article.get("link", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Format for our pipeline
            formatted = {
                "id": f"eodhd_{hash(url) & 0xFFFFFFFF:08x}",
                "title": article.get("title", ""),
                "text": article.get("content", ""),
                "url": url,
                "source": "EODHD",
                "source_type": "eodhd",
                "published_date": article.get("date", ""),
                "tickers": article.get("symbols", []),
                "tags": article.get("tags", []),
                "sentiment": article.get("sentiment", {}),
                "collected_at": datetime.now().isoformat(),
                "query_params": {
                    "ticker": ticker,
                    "topic": topic,
                    "from": from_date,
                    "to": to_date,
                }
            }
            all_articles.append(formatted)
            new_count += 1

        logger.info(f"  New unique: {new_count} articles")

        # Save incrementally
        with open(output_file, "a") as f:
            for article in all_articles[-new_count:]:
                f.write(json.dumps(article) + "\n")

        # Respect rate limits
        time.sleep(0.5)

    # Update state
    state["collected_urls"] = list(seen_urls)[-50000:]  # Keep last 50k URLs
    state["last_collection"] = datetime.now().isoformat()
    state["total_articles"] = state.get("total_articles", 0) + len(all_articles)
    save_collection_state(state)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"New articles collected: {len(all_articles)}")
    logger.info(f"Saved to: {output_file}")
    logger.info(f"Total articles (all time): {state['total_articles']}")

    if all_articles:
        # Date distribution
        dates = [a["published_date"][:10] for a in all_articles if a.get("published_date")]
        if dates:
            logger.info(f"Date range: {min(dates)} to {max(dates)}")

        # Source distribution
        ticker_counts = {}
        for a in all_articles:
            for t in a.get("tickers", []):
                ticker_counts[t] = ticker_counts.get(t, 0) + 1

        if ticker_counts:
            logger.info("\nTop tickers in collection:")
            for ticker, count in sorted(ticker_counts.items(), key=lambda x: -x[1])[:10]:
                logger.info(f"  {ticker}: {count}")

    return all_articles


def main():
    parser = argparse.ArgumentParser(
        description="EODHD News Collector - Optimized for Free Tier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  topics      - Collect by financial topics (earnings, M&A, tech, finance)
  tickers     - Collect by major tickers (AAPL, NVDA, MSFT, GOOGL)
  historical  - Collect by time periods (2020-2026 in quarters)
  hybrid      - Mixed approach for diversity (recommended)
  sector_rotation - Rotate through market sectors

Examples:
  python collect_eodhd_news.py                     # Use hybrid strategy
  python collect_eodhd_news.py --strategy topics   # Use topics strategy
  python collect_eodhd_news.py --dry-run           # Preview without executing
  python collect_eodhd_news.py --max-requests 2    # Limit to 2 requests
        """
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=list(STRATEGIES.keys()),
        help="Collection strategy (default: hybrid)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Maximum requests to use (default: all remaining)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing",
    )
    parser.add_argument(
        "--show-strategies",
        action="store_true",
        help="Show all available strategies and exit",
    )

    args = parser.parse_args()

    if args.show_strategies:
        print("\nAvailable Collection Strategies:")
        print("=" * 60)
        for name, strategy in STRATEGIES.items():
            print(f"\n{name}:")
            print(f"  {strategy['description']}")
            print("  Queries:")
            for q in strategy["queries"]:
                desc = q.get("description", "")
                params = {k: v for k, v in q.items() if k != "description"}
                print(f"    - {desc}: {params}")
        return

    # Get API key
    api_key = get_api_key()
    if not api_key:
        logger.error("No EODHD API key found!")
        logger.error("Set EODHD_API_KEY env var or add to config/secrets.yaml")
        sys.exit(1)

    # Check API status
    remaining = get_remaining_requests(api_key)
    logger.info(f"API requests remaining today: {remaining}")

    if remaining <= 0 and not args.dry_run:
        logger.warning("No API requests remaining today!")
        logger.info("Run with --dry-run to see the collection plan")
        logger.info("API resets at midnight UTC")
        sys.exit(0)

    # Execute collection
    articles = collect_with_strategy(
        api_key=api_key,
        strategy_name=args.strategy,
        max_requests=args.max_requests,
        dry_run=args.dry_run,
    )

    if articles:
        logger.info(f"\nReady for labeling: {len(articles)} articles")
        logger.info("Run: python scripts/labeling/label_with_haiku.py --source eodhd")


if __name__ == "__main__":
    main()
