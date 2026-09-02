"""NewsAPI Collector for Reuters and other financial news sources.

Requires a NewsAPI key from https://newsapi.org (free tier: 100 requests/day).
"""

import time
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


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


class NewsAPICollector:
    """
    Collect financial news from NewsAPI.

    NewsAPI provides access to news from Reuters, Bloomberg, CNBC, and more.
    Free tier: 100 requests/day, articles up to 1 month old.

    Get your API key at: https://newsapi.org/register

    Attributes:
        api_key: NewsAPI key
        base_url: API endpoint
    """

    BASE_URL = "https://newsapi.org/v2"

    # Financial news sources available on NewsAPI
    FINANCIAL_SOURCES = [
        "reuters",
        "bloomberg",
        "the-wall-street-journal",
        "cnbc",
        "financial-times",
        "fortune",
        "business-insider",
        "the-economist",
    ]

    # Financial keywords for searching
    FINANCIAL_KEYWORDS = [
        "stock", "shares", "earnings", "revenue", "profit",
        "quarterly results", "guidance", "analyst", "upgrade",
        "downgrade", "merger", "acquisition", "IPO", "dividend",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        sources: Optional[List[str]] = None,
        delay: float = 0.5,
    ):
        """
        Initialize NewsAPI collector.

        Args:
            api_key: NewsAPI key. If None, reads from NEWSAPI_KEY env var.
            sources: List of source IDs to query. Defaults to financial sources.
            delay: Delay between requests in seconds
        """
        if requests is None:
            raise ImportError("requests is required. Install with: pip install requests")

        self.api_key = api_key or os.environ.get("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError(
                "NewsAPI key required. Set NEWSAPI_KEY env var or pass api_key. "
                "Get a free key at: https://newsapi.org/register"
            )

        self.sources = sources or self.FINANCIAL_SOURCES
        self.delay = delay

    def _make_request(self, endpoint: str, params: dict) -> Optional[dict]:
        """
        Make a request to NewsAPI.

        Args:
            endpoint: API endpoint (e.g., "everything", "top-headlines")
            params: Query parameters

        Returns:
            JSON response or None if failed
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params["apiKey"] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                logger.error(f"API error: {data.get('message', 'Unknown error')}")
                return None

            return data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Invalid API key")
            elif e.response.status_code == 429:
                logger.error("Rate limit exceeded. Free tier: 100 requests/day")
            else:
                logger.error(f"HTTP error: {e}")
            return None

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def search(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        sources: Optional[List[str]] = None,
        page_size: int = 100,
        page: int = 1,
    ) -> List[Article]:
        """
        Search for articles matching a query.

        Args:
            query: Search query (e.g., "Apple earnings")
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            sources: Override default sources
            page_size: Results per page (max 100)
            page: Page number

        Returns:
            List of Article objects
        """
        params = {
            "q": query,
            "pageSize": min(page_size, 100),
            "page": page,
            "language": "en",
            "sortBy": "publishedAt",
        }

        # Add sources
        sources = sources or self.sources
        if sources:
            params["sources"] = ",".join(sources)

        # Add date range
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = self._make_request("everything", params)
        if not data:
            return []

        articles = []
        for item in data.get("articles", []):
            # Extract text (content is often truncated, use description if available)
            content = item.get("content", "")
            description = item.get("description", "")
            title = item.get("title", "")

            # Combine available text
            text_parts = [title]
            if description:
                text_parts.append(description)
            if content:
                # NewsAPI truncates content with "[+N chars]", clean it
                content = content.split("[+")[0].strip()
                if content and content not in description:
                    text_parts.append(content)

            text = ". ".join(text_parts)

            # Extract tickers from query (simple heuristic)
            tickers = self._extract_tickers_from_query(query)

            article = Article(
                title=title,
                text=text,
                url=item.get("url", ""),
                source=item.get("source", {}).get("name", "NewsAPI"),
                published_date=item.get("publishedAt", ""),
                tickers=tickers,
                collected_at=datetime.now().isoformat(),
            )
            articles.append(article)

        return articles

    def _extract_tickers_from_query(self, query: str) -> List[str]:
        """Extract potential ticker symbols from query."""
        # Common company name to ticker mappings
        company_to_ticker = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "meta": "META",
            "facebook": "META",
            "nvidia": "NVDA",
            "tesla": "TSLA",
            "jpmorgan": "JPM",
            "goldman": "GS",
            "netflix": "NFLX",
        }

        tickers = []
        query_lower = query.lower()

        for company, ticker in company_to_ticker.items():
            if company in query_lower:
                tickers.append(ticker)

        # Also check for uppercase ticker symbols in query
        for word in query.split():
            if word.isupper() and 1 <= len(word) <= 5:
                tickers.append(word)

        return list(set(tickers))

    def collect_by_companies(
        self,
        companies: List[str],
        days_back: int = 7,
        articles_per_company: int = 20,
    ) -> List[Article]:
        """
        Collect news for a list of companies.

        Args:
            companies: List of company names or tickers
            days_back: How many days back to search
            articles_per_company: Max articles per company

        Returns:
            List of Article objects
        """
        all_articles = []
        seen_urls = set()

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        for company in companies:
            logger.info(f"Collecting news for {company}...")

            articles = self.search(
                query=company,
                from_date=from_date,
                to_date=to_date,
                page_size=articles_per_company,
            )

            # Deduplicate
            for article in articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    # Add company/ticker to article
                    if company.upper() not in article.tickers:
                        article.tickers.append(company.upper())
                    all_articles.append(article)

            time.sleep(self.delay)

        logger.info(f"Collected {len(all_articles)} unique articles")
        return all_articles

    def collect_top_headlines(
        self,
        category: str = "business",
        country: str = "us",
        page_size: int = 100,
    ) -> List[Article]:
        """
        Collect top business headlines.

        Args:
            category: News category (business, technology, etc.)
            country: Country code
            page_size: Number of articles

        Returns:
            List of Article objects
        """
        params = {
            "category": category,
            "country": country,
            "pageSize": min(page_size, 100),
        }

        data = self._make_request("top-headlines", params)
        if not data:
            return []

        articles = []
        for item in data.get("articles", []):
            title = item.get("title", "")
            description = item.get("description", "")
            content = item.get("content", "").split("[+")[0].strip()

            text = ". ".join(filter(None, [title, description, content]))

            article = Article(
                title=title,
                text=text,
                url=item.get("url", ""),
                source=item.get("source", {}).get("name", "NewsAPI"),
                published_date=item.get("publishedAt", ""),
                tickers=[],  # Headlines don't have specific tickers
                collected_at=datetime.now().isoformat(),
            )
            articles.append(article)

        return articles

    def collect_financial_news(
        self,
        days_back: int = 7,
        max_articles: int = 500,
    ) -> List[Article]:
        """
        Collect general financial news using financial keywords.

        Args:
            days_back: How many days back to search
            max_articles: Maximum total articles

        Returns:
            List of Article objects
        """
        all_articles = []
        seen_urls = set()

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        for keyword in self.FINANCIAL_KEYWORDS:
            if len(all_articles) >= max_articles:
                break

            logger.info(f"Searching for '{keyword}'...")

            articles = self.search(
                query=keyword,
                from_date=from_date,
                page_size=50,
            )

            for article in articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)

            time.sleep(self.delay)

        logger.info(f"Collected {len(all_articles)} unique articles")
        return all_articles[:max_articles]

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
