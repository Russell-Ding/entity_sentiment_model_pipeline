"""Yahoo Finance News Collector.

Collects news articles for specified stock tickers using yfinance library.
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

try:
    import yfinance as yf
except (ImportError, TypeError):
    # TypeError can occur with Python < 3.10 due to type hint syntax
    yf = None

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None


logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Represents a news article."""
    title: str
    text: str
    url: str
    source: str
    published_date: str
    tickers: List[str]
    collected_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class YahooFinanceCollector:
    """
    Collect news articles from Yahoo Finance.

    Uses yfinance to get news metadata and optionally fetches full article text.

    Attributes:
        tickers: List of stock tickers to collect news for
        delay: Delay between requests in seconds
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        delay: float = 1.0,
        fetch_full_text: bool = True,
    ):
        """
        Initialize Yahoo Finance collector.

        Args:
            tickers: List of stock tickers (e.g., ["AAPL", "GOOGL"])
            delay: Delay between requests to avoid rate limiting
            fetch_full_text: Whether to fetch full article text (slower)
        """
        if yf is None:
            raise ImportError(
                "yfinance is required. Install with: pip install yfinance"
            )

        self.tickers = tickers or []
        self.delay = delay
        self.fetch_full_text = fetch_full_text

        # Default top stocks if none provided
        if not self.tickers:
            self.tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META",
                "NVDA", "TSLA", "JPM", "V", "JNJ",
                "WMT", "PG", "MA", "HD", "BAC",
                "DIS", "NFLX", "CRM", "AMD", "INTC",
            ]

    def _fetch_article_text(self, url: str) -> Optional[str]:
        """
        Fetch full article text from URL.

        Args:
            url: Article URL

        Returns:
            Article text or None if failed
        """
        if requests is None or BeautifulSoup is None:
            logger.warning("requests/beautifulsoup4 not installed, skipping full text fetch")
            return None

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Try common article containers
            article_selectors = [
                "article",
                '[class*="article-body"]',
                '[class*="article-content"]',
                '[class*="story-body"]',
                '[class*="post-content"]',
                "main",
            ]

            for selector in article_selectors:
                article = soup.select_one(selector)
                if article:
                    paragraphs = article.find_all("p")
                    if paragraphs:
                        text = " ".join(p.get_text().strip() for p in paragraphs)
                        if len(text) > 200:  # Minimum viable article length
                            return text

            # Fallback: get all paragraphs
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text().strip() for p in paragraphs)

            return text if len(text) > 200 else None

        except Exception as e:
            logger.debug(f"Failed to fetch article from {url}: {e}")
            return None

    def collect_for_ticker(self, ticker: str) -> List[Article]:
        """
        Collect news articles for a single ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of Article objects
        """
        articles = []

        try:
            stock = yf.Ticker(ticker)
            news_items = stock.news

            if not news_items:
                logger.info(f"No news found for {ticker}")
                return articles

            for item in news_items:
                title = item.get("title", "")
                url = item.get("link", "")
                source = item.get("publisher", "Yahoo Finance")

                # Convert timestamp
                pub_timestamp = item.get("providerPublishTime", 0)
                pub_date = datetime.fromtimestamp(pub_timestamp).isoformat()

                # Get related tickers
                related_tickers = item.get("relatedTickers", [ticker])
                if ticker not in related_tickers:
                    related_tickers.append(ticker)

                # Fetch full text if enabled
                if self.fetch_full_text and url:
                    time.sleep(self.delay)
                    text = self._fetch_article_text(url)
                else:
                    text = None

                # Use title + summary if no full text
                if not text:
                    summary = item.get("summary", "")
                    text = f"{title}. {summary}" if summary else title

                article = Article(
                    title=title,
                    text=text,
                    url=url,
                    source=source,
                    published_date=pub_date,
                    tickers=related_tickers,
                    collected_at=datetime.now().isoformat(),
                )
                articles.append(article)

        except Exception as e:
            logger.error(f"Error collecting news for {ticker}: {e}")

        return articles

    def collect(
        self,
        tickers: Optional[List[str]] = None,
        max_per_ticker: int = 10,
    ) -> List[Article]:
        """
        Collect news articles for all tickers.

        Args:
            tickers: Override default tickers
            max_per_ticker: Maximum articles per ticker

        Returns:
            List of all collected articles
        """
        tickers = tickers or self.tickers
        all_articles = []
        seen_urls = set()

        for ticker in tickers:
            logger.info(f"Collecting news for {ticker}...")
            articles = self.collect_for_ticker(ticker)

            # Deduplicate by URL
            for article in articles[:max_per_ticker]:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)

            time.sleep(self.delay)

        logger.info(f"Collected {len(all_articles)} unique articles")
        return all_articles

    def save(
        self,
        articles: List[Article],
        output_path: str,
        format: str = "jsonl",
    ) -> None:
        """
        Save collected articles to file.

        Args:
            articles: List of articles to save
            output_path: Output file path
            format: Output format ("jsonl" or "json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for article in articles:
                    f.write(json.dumps(article.to_dict()) + "\n")
        else:
            with open(output_path, "w") as f:
                json.dump([a.to_dict() for a in articles], f, indent=2)

        logger.info(f"Saved {len(articles)} articles to {output_path}")
