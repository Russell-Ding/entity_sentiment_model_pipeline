#!/usr/bin/env python3
"""EODHD S&P 500 Bulk Collector — News + Prices (2020-2025).

Design review fixes incorporated:
- Date-window slicing with adaptive 6-month → 3-month fallback to avoid silent
  offset caps.
- Shared token-bucket rate limiter across workers (respects Retry-After).
- Plain-text temp files per ticker, gzip-finalized on completion (no append-gzip
  corruption risk).
- URL canonicalization (strip tracking params + fragments) for stable IDs and
  within-ticker dedup.
- Atomic progress.json writes; SIGTERM/SIGINT graceful shutdown.
- Pre-flight API quota and disk-space checks.
- Auto-updates data/collection_tracking.json on completion.

Usage:
    caffeinate -ims python scripts/collection/collect_eodhd_sp500_bulk.py \
        --tickers-csv data/raw/sp500_components_20260517.csv \
        --workers 8 \
        --rate-limit 1.0
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import random
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TICKERS_CSV = PROJECT_ROOT / "data" / "raw" / "sp500_components_20260517.csv"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "data" / "raw"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EODHD_NEWS_URL = "https://eodhd.com/api/news"
EODHD_EOD_URL = "https://eodhd.com/api/eod"
EODHD_USER_URL = "https://eodhd.com/api/user"

# Each EODHD HTTP request decrements daily quota by exactly 1, verified via
# X-RateLimit-Remaining headers on /api/news and /api/eod on 2026-05-17.
# (Earlier review claimed 5× cost for news — that was wrong.)
NEWS_API_COST = 1
PRICE_API_COST = 1

DATE_FROM = "2020-01-01"
DATE_TO = "2025-12-31"
TRAINING_DATE_MIN = "2020-10-02"
TRAINING_DATE_MAX = "2026-01-19"

MAX_OFFSET = 5000  # Safety cap before we split the window
LIMIT = 1000

# Query params stripped during URL canonicalization
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "source", "fbclid", "gclid", "ttclid", "irclickid",
    "cmpid", "ncid", "mod", "cli", "src", "affiliate",
    "tsrc", ".tsrc", "rf", "referrer", "tracking", "track", "cid",
}

# Fields from EODHD response mapped to our schema
EODHD_SENTIMENT_FIELDS = {"polarity", "neg", "neu", "pos"}

# ---------------------------------------------------------------------------
# Source detection & tiering
# ---------------------------------------------------------------------------
WIRE_PREFIX_RE = re.compile(
    r"^\s*(?:\((Bloomberg(?:\s+Opinion)?|Reuters|WSJ|FT|AP|AFP)\)"
    r"|(?:--\s*)?(Bloomberg|Reuters|WSJ|FT|AP|AFP)\s*--)",
    re.IGNORECASE,
)

BODY_ATTRIBUTION_RE = re.compile(
    r"(Zacks\s+Investment\s+Research|Zacks\s+Equity\s+Research|Benzinga|"
    r"GlobeNewswire|PR\s*Newswire|Business\s+Wire|MarketWatch)",
    re.IGNORECASE,
)

DOMAIN_TO_SOURCE: Dict[str, str] = {
    "finance.yahoo.com": "Yahoo Finance",
    "yahoo.com": "Yahoo Finance",
    "seekingalpha.com": "Seeking Alpha",
    "cnbc.com": "CNBC",
    "marketwatch.com": "MarketWatch",
    "nasdaq.com": "NASDAQ",
    "investing.com": "Investing.com",
    "globenewswire.com": "GlobeNewswire",
    "fool.com": "Motley Fool",
    "fxstreet.com": "FXStreet",
    "benzinga.com": "Benzinga",
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "wsj.com": "WSJ",
    "ft.com": "FT",
    "ap.org": "AP",
    "afp.com": "AFP",
    "investorplace.com": "InvestorPlace",
    "zacks.com": "Zacks",
}

SOURCE_TIER_MAP: Dict[str, int] = {
    # Tier 1: premium / wire / analyst-grade
    "Bloomberg": 1,
    "Bloomberg Opinion": 1,
    "Reuters": 1,
    "WSJ": 1,
    "FT": 1,
    "AP": 1,
    "AFP": 1,
    "MarketWatch": 1,
    "CNBC": 1,
    "Seeking Alpha": 1,
    # Tier 2: mixed quality / syndication / decent data
    "NASDAQ": 2,
    "Investing.com": 2,
    "GlobeNewswire": 2,
    "Motley Fool": 2,
    "FXStreet": 2,
    "Benzinga": 2,
    # Tier 3: default / low signal / SEO / promotional / algo-generated
    "Yahoo Finance": 3,
    "InvestorPlace": 3,
    # Zacks brands are all algo-generated stock bulletins ("X closed at $Y,
    # moving Z%") and SEO templates ("3 dividend stocks to consider"). Not
    # useful as investment signal regardless of which Zacks brand is cited.
    "Zacks Investment Research": 3,
    "Zacks Equity Research": 3,
    "Zacks": 3,
    "PR Newswire": 3,
    "Business Wire": 3,
}


def detect_source(title: str, content: str, url_domain: str) -> str:
    """Detect content source from attribution patterns or domain fallback.

    Body-attribution search is restricted to the first 1000 chars (title +
    content prefix) to avoid false positives from footer disclaimers like
    "powered by Zacks Investment Research" or "MarketWatch" ad blocks
    appended at the end of Yahoo articles. Must stay consistent with
    apply_source_tiers.py:apply_rules.
    """
    # 1. Wire prefixes at start of content (anchored to ^)
    if content:
        m = WIRE_PREFIX_RE.match(content)
        if m:
            return (m.group(1) or m.group(2)).strip()

    # 2. Body attribution patterns — only search first 1000 chars
    text = f"{title}\n{content}" if title or content else ""
    if text:
        m = BODY_ATTRIBUTION_RE.search(text[:1000])
        if m:
            return m.group(1).strip()

    # 3. Domain fallback
    clean_domain = url_domain.lower()
    if clean_domain.startswith("www."):
        clean_domain = clean_domain[4:]
    return DOMAIN_TO_SOURCE.get(clean_domain, clean_domain)


def assign_source_tier(detected_source: str) -> int:
    """Return static tier (1/2/3) for a detected source."""
    return SOURCE_TIER_MAP.get(detected_source, 3)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("eodhd_bulk")


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(threadName)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------------
# Shutdown handling
# ---------------------------------------------------------------------------
_shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Config / API key
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    key = os.environ.get("EODHD_API_KEY")
    if key:
        return key
    secrets_path = PROJECT_ROOT / "config" / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path) as f:
            data = yaml.safe_load(f) or {}
        key = data.get("api", {}).get("eodhd_api")
        if key:
            return key
    raise SystemExit("EODHD_API_KEY not found in env or config/secrets.yaml")


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class TokenBucketRateLimiter:
    """Thread-safe token bucket. Tokens added at `rate` per second."""

    def __init__(self, rate: float, burst: int = 5):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_update
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last_update = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                # Calculate sleep needed
                needed = tokens - self._tokens
                sleep_time = needed / self.rate
            if deadline and time.monotonic() + sleep_time > deadline:
                return False
            time.sleep(min(sleep_time, 0.1))


# ---------------------------------------------------------------------------
# API helpers with retry
# ---------------------------------------------------------------------------
def _backoff_sleep(attempt: int) -> float:
    """Exponential backoff: 1s, 4s, 16s, 60s, 120s."""
    delays = [1.0, 4.0, 16.0, 60.0, 120.0]
    return delays[min(attempt, len(delays) - 1)]


def _parse_retry_after(response: requests.Response) -> Optional[float]:
    val = response.headers.get("Retry-After")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return None


def fetch_news_page(
    session: requests.Session,
    api_key: str,
    symbol: str,
    from_date: str,
    to_date: str,
    offset: int,
    limit: int = LIMIT,
    max_retries: int = 5,
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    """Fetch one news page. Returns (articles, success, metadata).

    metadata includes api_calls_used and rate_limit info.
    """
    params = {
        "s": symbol,
        "from": from_date,
        "to": to_date,
        "offset": offset,
        "limit": limit,
        "api_token": api_key,
        "fmt": "json",
    }
    metadata = {"api_calls_used": NEWS_API_COST, "rate_limited": False}

    for attempt in range(max_retries):
        if _shutdown_event.is_set():
            return [], False, metadata
        try:
            resp = session.get(EODHD_NEWS_URL, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Request exception (attempt {attempt + 1}): {exc}")
            time.sleep(_backoff_sleep(attempt))
            continue

        if resp.status_code == 429:
            retry_after = _parse_retry_after(resp) or _backoff_sleep(attempt)
            logger.warning(f"Rate limited (429). Sleeping {retry_after:.1f}s ...")
            metadata["rate_limited"] = True
            time.sleep(retry_after)
            continue

        if resp.status_code >= 500:
            logger.warning(f"Server error {resp.status_code} (attempt {attempt + 1})")
            time.sleep(_backoff_sleep(attempt))
            continue

        if resp.status_code != 200:
            logger.error(f"HTTP {resp.status_code} for {symbol}: {resp.text[:200]}")
            return [], False, metadata

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning(f"JSON decode error (attempt {attempt + 1}): {exc}")
            time.sleep(_backoff_sleep(attempt))
            continue

        if not isinstance(data, list):
            logger.error(f"Unexpected response type: {type(data)}")
            return [], False, metadata

        # Update metadata with rate limit headers if present
        metadata["x_ratelimit_remaining"] = resp.headers.get("X-RateLimit-Remaining")
        return data, True, metadata

    logger.error(f"Max retries exceeded for news page {symbol} offset={offset}")
    return [], False, metadata


def fetch_price_history(
    session: requests.Session,
    api_key: str,
    symbol: str,
    max_retries: int = 5,
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    """Fetch full EOD price history. Returns (rows, success, metadata)."""
    url = f"{EODHD_EOD_URL}/{symbol}"
    params = {"api_token": api_key, "fmt": "json"}
    metadata = {"api_calls_used": PRICE_API_COST}

    for attempt in range(max_retries):
        if _shutdown_event.is_set():
            return [], False, metadata
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Price request exception (attempt {attempt + 1}): {exc}")
            time.sleep(_backoff_sleep(attempt))
            continue

        if resp.status_code == 429:
            retry_after = _parse_retry_after(resp) or _backoff_sleep(attempt)
            logger.warning(f"Price rate limited (429). Sleeping {retry_after:.1f}s ...")
            time.sleep(retry_after)
            continue

        if resp.status_code >= 500:
            logger.warning(f"Price server error {resp.status_code}")
            time.sleep(_backoff_sleep(attempt))
            continue

        if resp.status_code != 200:
            logger.error(f"Price HTTP {resp.status_code} for {symbol}")
            return [], False, metadata

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning(f"Price JSON decode error: {exc}")
            time.sleep(_backoff_sleep(attempt))
            continue

        if not isinstance(data, list):
            logger.error(f"Price unexpected type: {type(data)}")
            return [], False, metadata

        metadata["x_ratelimit_remaining"] = resp.headers.get("X-RateLimit-Remaining")
        return data, True, metadata

    logger.error(f"Max retries exceeded for price {symbol}")
    return [], False, metadata


def check_api_quota(session: requests.Session, api_key: str) -> Dict[str, Any]:
    """Return current API usage stats."""
    try:
        resp = session.get(
            EODHD_USER_URL, params={"api_token": api_key, "fmt": "json"}, timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning(f"Could not check API quota: {exc}")
        return {}


# ---------------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------------
def canonicalize_url(url: str) -> str:
    """Strip tracking params and fragments for stable dedup hashing."""
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qsl(parsed.query)
    filtered = [(k, v) for k, v in qs if k.lower() not in TRACKING_PARAMS]
    new_query = urllib.parse.urlencode(filtered)
    # Rebuild without fragment
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, "")
    )


def stable_id(url: str) -> str:
    """Stable 12-hex-char ID from canonical URL."""
    return hashlib.sha256(canonicalize_url(url).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Date / window utilities
# ---------------------------------------------------------------------------
def _date_obj(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def _date_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def generate_windows(from_date: str, to_date: str, months: int = 6) -> List[Tuple[str, str]]:
    """Generate non-overlapping date windows of `months` length."""
    windows: List[Tuple[str, str]] = []
    start = _date_obj(from_date)
    end = _date_obj(to_date)
    while start <= end:
        # End of this window
        next_start = (start.replace(day=1) + timedelta(days=32))
        next_start = next_start.replace(day=1)
        # Actually, simpler: add months manually
        month = start.month + months - 1
        year = start.year + month // 12
        month = month % 12 + 1
        window_end = datetime(year, month, 1) - timedelta(days=1)
        if window_end > end:
            window_end = end
        windows.append((_date_str(start), _date_str(window_end)))
        start = window_end + timedelta(days=1)
    return windows


def split_window(from_date: str, to_date: str) -> List[Tuple[str, str]]:
    """Split a date range in half (for adaptive fallback)."""
    start = _date_obj(from_date)
    end = _date_obj(to_date)
    delta = (end - start).days
    mid = start + timedelta(days=delta // 2)
    return [
        (_date_str(start), _date_str(mid)),
        (_date_str(mid + timedelta(days=1)), _date_str(end)),
    ]


# ---------------------------------------------------------------------------
# Progress tracking (atomic writes)
# ---------------------------------------------------------------------------
class ProgressTracker:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = {
            "completed": [],
            "failed": [],
            "skipped": [],
            "api_calls_used": 0,
            "articles_total": 0,
            "started_at": datetime.now().astimezone().isoformat(),
            "last_update": None,
        }
        if path.exists():
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def record_completed(self, symbol: str, articles: int, api_calls: int):
        with self.lock:
            self.data["completed"].append(symbol)
            self.data["articles_total"] += articles
            self.data["api_calls_used"] += api_calls
            self.data["last_update"] = datetime.now().astimezone().isoformat()
            self._flush()

    def record_failed(self, symbol: str, reason: str, api_calls: int):
        with self.lock:
            self.data["failed"].append({"ticker": symbol, "reason": reason})
            self.data["api_calls_used"] += api_calls
            self.data["last_update"] = datetime.now().astimezone().isoformat()
            self._flush()

    def record_skipped(self, symbol: str, reason: str):
        with self.lock:
            self.data["skipped"].append({"ticker": symbol, "reason": reason})
            self.data["last_update"] = datetime.now().astimezone().isoformat()
            self._flush()

    def is_completed(self, symbol: str) -> bool:
        with self.lock:
            return symbol in self.data["completed"]

    def _flush(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(str(tmp), str(self.path))


# ---------------------------------------------------------------------------
# Record formatting
# ---------------------------------------------------------------------------
def format_article(raw: Dict[str, Any], symbol: str, collected_at: str) -> Optional[Dict[str, Any]]:
    """Convert EODHD article to our schema. Returns None if invalid."""
    url = raw.get("link", "")
    if not url:
        return None
    title = raw.get("title", "")
    content = raw.get("content", "")
    if not title and not content:
        return None

    date = raw.get("date", "")
    date_short = date[:10] if date else ""
    date_in_training_window = (
        TRAINING_DATE_MIN <= date_short <= TRAINING_DATE_MAX if date_short else False
    )

    eodhd_sentiment = raw.get("sentiment", {})
    # Validate shape
    if not isinstance(eodhd_sentiment, dict):
        eodhd_sentiment = {}
    else:
        eodhd_sentiment = {k: v for k, v in eodhd_sentiment.items() if k in EODHD_SENTIMENT_FIELDS}

    # --- Source & quality fields ---
    url_canonical = canonicalize_url(url)
    url_domain = urllib.parse.urlparse(url_canonical).netloc.lower()
    content_length = len(content) if content else 0
    is_truncated = content.rstrip().endswith("[...truncated]") if content else False
    symbols = raw.get("symbols", [])
    symbols_count = len(symbols) if isinstance(symbols, list) else 0

    detected_source = detect_source(title, content, url_domain)
    source_tier = assign_source_tier(detected_source)

    return {
        "id": stable_id(url),
        "url": url,
        "url_canonical": url_canonical,
        "url_domain": url_domain,
        "date": date,
        "title": title,
        "content": content,
        "content_length": content_length,
        "is_truncated": is_truncated,
        "symbols": symbols,
        "symbols_count": symbols_count,
        "tags": raw.get("tags", []),
        "eodhd_sentiment": eodhd_sentiment,
        "date_in_training_window": date_in_training_window,
        "primary_ticker": symbol,
        "source": "EODHD",
        "detected_source": detected_source,
        "source_tier": source_tier,
        "collected_at": collected_at,
    }


# ---------------------------------------------------------------------------
# News collection for a single ticker (adaptive windows)
# ---------------------------------------------------------------------------
def collect_ticker_news(
    session: requests.Session,
    rate_limiter: TokenBucketRateLimiter,
    api_key: str,
    symbol: str,
    from_date: str,
    to_date: str,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """Collect all news for one ticker using adaptive windows.

    Returns:
        (articles, api_calls_used, status_msg)
    """
    api_calls = 0
    windows: List[Tuple[str, str]] = [(_date_str(_date_obj(from_date)), _date_str(_date_obj(to_date)))]
    # We'll use 6-month windows as the baseline, splitting only when needed.
    # Override: start with a single full-range window and let the splitting
    # logic create sub-windows as needed.
    windows = generate_windows(from_date, to_date, months=6)
    all_articles: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()

    while windows:
        if _shutdown_event.is_set():
            return all_articles, api_calls, "shutdown"

        w_from, w_to = windows.pop(0)
        offset = 0
        window_articles: List[Dict[str, Any]] = []
        hit_cap = False

        while True:
            rate_limiter.acquire()
            page, ok, meta = fetch_news_page(
                session, api_key, symbol, w_from, w_to, offset, LIMIT
            )
            api_calls += meta["api_calls_used"]

            if not ok:
                return all_articles, api_calls, f"fetch_failed_at_offset_{offset}"

            if not page:
                break

            for raw in page:
                rec = format_article(raw, symbol, datetime.now().astimezone().isoformat())
                if rec is None:
                    continue
                if rec["url_canonical"] in seen_urls:
                    continue
                seen_urls.add(rec["url_canonical"])
                window_articles.append(rec)

            if len(page) < LIMIT:
                break

            offset += LIMIT
            if offset >= MAX_OFFSET:
                hit_cap = True
                break

        if hit_cap:
            # Window too dense — split and re-queue
            logger.info(f"{symbol}: window {w_from}→{w_to} dense (offset≥{MAX_OFFSET}), splitting")
            sub_windows = split_window(w_from, w_to)
            # Insert at front so we process them next
            for sw in reversed(sub_windows):
                windows.insert(0, sw)
            # Discard partial window results; they will be re-fetched in sub-windows
            for rec in window_articles:
                seen_urls.discard(rec["url_canonical"])
            continue

        all_articles.extend(window_articles)

    return all_articles, api_calls, "success"


# ---------------------------------------------------------------------------
# Price collection for a single ticker
# ---------------------------------------------------------------------------
def collect_ticker_prices(
    session: requests.Session,
    rate_limiter: TokenBucketRateLimiter,
    api_key: str,
    symbol: str,
) -> Tuple[List[Dict[str, Any]], int, str]:
    rate_limiter.acquire()
    rows, ok, meta = fetch_price_history(session, api_key, symbol)
    if not ok:
        return [], meta["api_calls_used"], "price_fetch_failed"
    return rows, meta["api_calls_used"], "success"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def worker_process_ticker(
    symbol: str,
    api_key: str,
    rate_limiter: TokenBucketRateLimiter,
    news_dir: Path,
    prices_dir: Path,
    progress: ProgressTracker,
) -> Dict[str, Any]:
    """Process one ticker (news + prices). Thread-safe."""
    if progress.is_completed(symbol):
        return {"symbol": symbol, "status": "skipped", "reason": "already_completed"}

    session = requests.Session()
    total_api_calls = 0

    # ---------- Prices ----------
    price_path = prices_dir / f"{symbol}.csv"
    if not price_path.exists():
        rows, calls, status = collect_ticker_prices(session, rate_limiter, api_key, symbol)
        total_api_calls += calls
        if status != "success":
            progress.record_failed(symbol, f"price:{status}", total_api_calls)
            return {"symbol": symbol, "status": "failed", "reason": status}
        # Write CSV
        if rows:
            import csv as csv_mod
            fieldnames = list(rows[0].keys())
            with open(price_path, "w", newline="") as f:
                writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"{symbol}: wrote {len(rows)} price rows")
    else:
        logger.info(f"{symbol}: price CSV already exists, skipping")

    # ---------- News ----------
    gz_path = news_dir / f"{symbol}.jsonl.gz"
    if gz_path.exists():
        progress.record_skipped(symbol, "news_gz_exists")
        return {"symbol": symbol, "status": "skipped", "reason": "news_gz_exists"}

    tmp_path = news_dir / f"{symbol}.jsonl.tmp"
    # If tmp exists from a crashed run, delete it to start fresh
    if tmp_path.exists():
        logger.warning(f"{symbol}: removing stale tmp file")
        tmp_path.unlink()

    articles, calls, status = collect_ticker_news(
        session, rate_limiter, api_key, symbol, DATE_FROM, DATE_TO
    )
    total_api_calls += calls

    if status != "success":
        progress.record_failed(symbol, f"news:{status}", total_api_calls)
        return {"symbol": symbol, "status": "failed", "reason": status}

    # Write to temp plain JSONL, then gzip atomically
    with open(tmp_path, "w") as f:
        for rec in articles:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(tmp_path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    tmp_path.unlink()
    progress.record_completed(symbol, len(articles), total_api_calls)
    logger.info(f"{symbol}: completed — {len(articles)} articles, {total_api_calls} API calls")
    return {
        "symbol": symbol,
        "status": "success",
        "articles": len(articles),
        "api_calls": total_api_calls,
    }


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def preflight_checks(
    session: requests.Session,
    api_key: str,
    news_dir: Path,
    prices_dir: Path,
    tickers: List[str],
    rate_limit: float = 4.0,
) -> bool:
    """Return True if safe to proceed."""
    # 1. API quota
    usage = check_api_quota(session, api_key)
    daily_limit = usage.get("dailyRateLimit", 100_000)
    used = usage.get("apiRequests", 0)
    remaining = daily_limit - used
    # Rough estimate: avg ~50 articles/month × 72 months = ~3,600 articles per
    # ticker → ~4 pages = 4 news calls. AAPL-class tickers go higher (~150
    # calls) but they're a minority. Plus 1 price call. Budget ~60/ticker
    # which keeps us well under the 100K/day limit even if some tickers
    # blow up.
    needed = len(tickers) * 60
    logger.info(f"API quota: {used:,} used / {daily_limit:,} limit. Remaining: {remaining:,}")
    logger.info(f"Estimated need: ~{needed:,} API calls")
    hrs_at_rate = needed / (rate_limit * 3600) if rate_limit else None
    if hrs_at_rate:
        logger.info(f"Estimated wall-clock at {rate_limit:.1f} req/sec: ~{hrs_at_rate:.1f} hours")
    if remaining < needed:
        logger.error(f"Insufficient API quota. Need ~{needed}, have {remaining}. Aborting.")
        return False

    # 2. Output dirs (must exist before disk_usage)
    news_dir.mkdir(parents=True, exist_ok=True)
    prices_dir.mkdir(parents=True, exist_ok=True)

    # 3. Disk space
    free = shutil.disk_usage(news_dir).free
    needed_bytes = 60 * 1024**3  # 60 GB conservative
    logger.info(f"Disk free: {free / 1024**3:.1f} GB")
    if free < needed_bytes:
        logger.error(f"Insufficient disk space. Need ~{needed_bytes / 1024**3:.0f} GB, have {free / 1024**3:.1f} GB.")
        return False

    return True


# ---------------------------------------------------------------------------
# Update collection_tracking.json
# ---------------------------------------------------------------------------
def update_collection_tracking(
    news_dir: Path,
    prices_dir: Path,
    progress: ProgressTracker,
    started_at: str,
    finished_at: str,
) -> None:
    tracking_path = PROJECT_ROOT / "data" / "collection_tracking.json"
    tracking: Dict[str, Any] = {}
    if tracking_path.exists():
        try:
            with open(tracking_path) as f:
                tracking = json.load(f)
        except Exception:
            tracking = {}

    if "sources" not in tracking:
        tracking["sources"] = {}

    # Count files and articles
    news_files = sorted(news_dir.glob("*.jsonl.gz"))
    prices_files = sorted(prices_dir.glob("*.csv"))
    total_articles = 0
    date_min = None
    date_max = None
    for nf in news_files:
        # Count lines without full decompress
        with gzip.open(nf, "rt") as f:
            for line in f:
                if line.strip():
                    total_articles += 1
                    try:
                        rec = json.loads(line)
                        d = rec.get("date", "")[:10]
                        if d:
                            if date_min is None or d < date_min:
                                date_min = d
                            if date_max is None or d > date_max:
                                date_max = d
                    except Exception:
                        pass

    run_key = f"eodhd_sp500_bulk_{datetime.now().strftime('%Y%m%d')}"
    tracking["sources"][run_key] = {
        "files": [f.name for f in news_files],
        "price_files": [f.name for f in prices_files],
        "total_articles": total_articles,
        "date_range": {"min": date_min, "max": date_max},
        "tickers_completed": progress.data["completed"],
        "tickers_failed": progress.data["failed"],
        "api_calls_used": progress.data["api_calls_used"],
        "started_at": started_at,
        "finished_at": finished_at,
        "contamination_status": "date_overlap_with_training_window",
    }

    tmp = tracking_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(tracking, f, indent=2)
    os.replace(str(tmp), str(tracking_path))
    logger.info(f"Updated {tracking_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EODHD S&P 500 Bulk Collector")
    parser.add_argument(
        "--tickers-csv",
        type=Path,
        default=DEFAULT_TICKERS_CSV,
        help="CSV with S&P 500 components (must have 'eodhd_symbol' column)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory (default: data/raw/eodhd_news_YYYYMMDD)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=4.0,
        help="Max HTTP requests per second across all workers (default: 4.0)",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Limit to first N tickers (for testing)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle ticker order (useful for resuming with partial progress)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan and pre-flight checks without executing API calls",
    )
    args = parser.parse_args()

    # Load tickers
    if not args.tickers_csv.exists():
        raise SystemExit(f"Tickers CSV not found: {args.tickers_csv}")
    with open(args.tickers_csv) as f:
        reader = csv.DictReader(f)
        tickers = [row["eodhd_symbol"] for row in reader if row.get("eodhd_symbol")]
    if not tickers:
        raise SystemExit("No tickers loaded from CSV")
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
    if args.shuffle:
        random.shuffle(tickers)
    logger.info(f"Loaded {len(tickers)} tickers")

    # Output dirs
    today = datetime.now().strftime("%Y%m%d")
    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = DEFAULT_OUTPUT_BASE / f"eodhd_bulk_{today}"
    news_dir = base_dir / "news"
    prices_dir = base_dir / "prices"
    progress_path = news_dir / "_progress.json"

    # Logging
    log_path = LOG_DIR / f"eodhd_collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_path)
    logger.info("=" * 60)
    logger.info("EODHD S&P 500 Bulk Collector starting")
    logger.info(f"Tickers: {len(tickers)} | Workers: {args.workers} | Rate limit: {args.rate_limit}/sec")
    logger.info(f"Output: {base_dir}")
    logger.info(f"Log: {log_path}")
    logger.info("=" * 60)

    # API key
    api_key = load_api_key()
    logger.info(f"API key loaded: {api_key[:4]}...{api_key[-4:]}")

    # Pre-flight
    session = requests.Session()
    if not preflight_checks(session, api_key, news_dir, prices_dir, tickers, rate_limit=args.rate_limit):
        raise SystemExit("Preflight checks failed")

    if args.dry_run:
        logger.info("[DRY RUN] No API calls will be made. Exiting.")
        logger.info(f"Would process {len(tickers)} tickers with {args.workers} workers")
        logger.info(f"Output directory: {base_dir}")
        sys.exit(0)

    # Progress
    progress = ProgressTracker(progress_path)
    already_done = set(progress.data.get("completed", []))
    logger.info(f"Resuming: {len(already_done)} already completed, {len(tickers) - len(already_done)} remaining")

    # Rate limiter
    rate_limiter = TokenBucketRateLimiter(rate=args.rate_limit, burst=5)

    started_at = datetime.now().astimezone().isoformat()
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for sym in tickers:
            if progress.is_completed(sym):
                continue
            fut = executor.submit(
                worker_process_ticker,
                sym,
                api_key,
                rate_limiter,
                news_dir,
                prices_dir,
                progress,
            )
            futures[fut] = sym

        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as exc:
                logger.exception(f"{sym}: unhandled exception in worker")
                progress.record_failed(sym, f"exception:{exc}", 0)
                results.append({"symbol": sym, "status": "failed", "reason": str(exc)})

            if _shutdown_event.is_set():
                logger.warning("Shutdown requested, cancelling remaining futures...")
                for f in futures:
                    f.cancel()
                break

    finished_at = datetime.now().astimezone().isoformat()

    # Summary
    success = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    total_articles = sum(r.get("articles", 0) for r in results if r.get("status") == "success")

    logger.info("=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Success : {success}")
    logger.info(f"Failed  : {failed}")
    logger.info(f"Skipped : {skipped}")
    logger.info(f"Articles: {total_articles:,}")
    logger.info(f"API calls used (tracked): {progress.data['api_calls_used']:,}")
    logger.info(f"Progress file: {progress_path}")

    # Update collection tracking
    try:
        update_collection_tracking(news_dir, prices_dir, progress, started_at, finished_at)
    except Exception:
        logger.exception("Failed to update collection_tracking.json")

    if failed:
        logger.warning(f"{failed} tickers failed — review progress file and re-run to retry")


if __name__ == "__main__":
    main()
