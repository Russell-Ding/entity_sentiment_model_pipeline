"""Unified Data Collection Pipeline.

Combines Yahoo Finance, NewsAPI (Reuters), and SEC EDGAR collectors
into a single pipeline for collecting financial news articles.
"""

import logging
import sys
from datetime import datetime
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .yahoo_finance import YahooFinanceCollector, Article as YFArticle
except (ImportError, TypeError):
    YahooFinanceCollector = None
    YFArticle = None

from .newsapi_collector import NewsAPICollector, Article as NewsArticle
from .sec_edgar import SECEdgarCollector, Filing

try:
    from config import get_config, get_newsapi_key
except ImportError:
    # Fallback if config not available
    def get_config():
        return None
    def get_newsapi_key():
        return None


logger = logging.getLogger(__name__)


@dataclass
class UnifiedArticle:
    """Unified article format across all sources."""
    id: str
    title: str
    text: str
    url: str
    source: str
    source_type: str  # "yahoo_finance", "newsapi", "sec_edgar"
    published_date: str
    tickers: List[str]
    entities: List[str]  # Company names and entities
    collected_at: str
    metadata: Dict  # Source-specific metadata

    def to_dict(self) -> dict:
        return asdict(self)


class DataCollectionPipeline:
    """
    Unified pipeline for collecting financial news from multiple sources.

    Combines:
    - Yahoo Finance: Stock-specific news
    - NewsAPI: Reuters and other financial news
    - SEC EDGAR: Official company filings (8-K)

    Example:
        pipeline = DataCollectionPipeline(newsapi_key="your-key")
        articles = pipeline.collect(
            tickers=["AAPL", "MSFT", "GOOGL"],
            sources=["yahoo_finance", "newsapi", "sec_edgar"],
        )
        pipeline.save(articles, "data/articles.jsonl")
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        delay: float = 1.0,
    ):
        """
        Initialize the data collection pipeline.

        Args:
            newsapi_key: API key for NewsAPI. If None, loads from config/secrets.yaml.
                        Get free key at: https://newsapi.org/register
            user_agent: User agent for SEC EDGAR requests.
            delay: Delay between requests in seconds.
        """
        # Load from config if not provided
        config = get_config()

        if newsapi_key:
            self.newsapi_key = newsapi_key
        elif config and config.api.newsapi_key:
            self.newsapi_key = config.api.newsapi_key
        else:
            self.newsapi_key = get_newsapi_key()

        if config and config.collection:
            self.user_agent = user_agent or config.collection.sec_user_agent
            self.delay = delay or config.collection.delay
        else:
            self.user_agent = user_agent or "PersonalResearch contact@example.com"
            self.delay = delay

        # Lazy-loaded collectors
        self._yahoo_collector = None
        self._newsapi_collector = None
        self._edgar_collector = None

    @property
    def yahoo_collector(self):
        """Get or create Yahoo Finance collector."""
        if YahooFinanceCollector is None:
            raise ImportError(
                "Yahoo Finance collector not available. "
                "Requires Python 3.10+ or install yfinance: pip install yfinance"
            )
        if self._yahoo_collector is None:
            self._yahoo_collector = YahooFinanceCollector(
                delay=self.delay,
                fetch_full_text=True,
            )
        return self._yahoo_collector

    @property
    def newsapi_collector(self) -> NewsAPICollector:
        """Get or create NewsAPI collector."""
        if self._newsapi_collector is None:
            if not self.newsapi_key:
                raise ValueError(
                    "NewsAPI key required. Set newsapi_key parameter or "
                    "NEWSAPI_KEY environment variable. "
                    "Get free key at: https://newsapi.org/register"
                )
            self._newsapi_collector = NewsAPICollector(
                api_key=self.newsapi_key,
                delay=self.delay,
            )
        return self._newsapi_collector

    @property
    def edgar_collector(self) -> SECEdgarCollector:
        """Get or create SEC EDGAR collector."""
        if self._edgar_collector is None:
            self._edgar_collector = SECEdgarCollector(
                user_agent=self.user_agent,
                delay=0.1,  # SEC allows 10 req/sec
            )
        return self._edgar_collector

    def _generate_id(self, source: str, url: str) -> str:
        """Generate unique article ID."""
        import hashlib
        content = f"{source}:{url}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _convert_yahoo_article(self, article: YFArticle) -> UnifiedArticle:
        """Convert Yahoo Finance article to unified format."""
        return UnifiedArticle(
            id=self._generate_id("yahoo_finance", article.url),
            title=article.title,
            text=article.text,
            url=article.url,
            source=article.source,
            source_type="yahoo_finance",
            published_date=article.published_date,
            tickers=article.tickers,
            entities=article.tickers,  # Tickers as initial entities
            collected_at=article.collected_at,
            metadata={},
        )

    def _convert_newsapi_article(self, article: NewsArticle) -> UnifiedArticle:
        """Convert NewsAPI article to unified format."""
        return UnifiedArticle(
            id=self._generate_id("newsapi", article.url),
            title=article.title,
            text=article.text,
            url=article.url,
            source=article.source,
            source_type="newsapi",
            published_date=article.published_date,
            tickers=article.tickers,
            entities=article.tickers,
            collected_at=article.collected_at,
            metadata={},
        )

    def _convert_edgar_filing(self, filing: Filing) -> UnifiedArticle:
        """Convert SEC EDGAR filing to unified format."""
        return UnifiedArticle(
            id=self._generate_id("sec_edgar", filing.url),
            title=filing.title,
            text=filing.text,
            url=filing.url,
            source=filing.source,
            source_type="sec_edgar",
            published_date=filing.published_date,
            tickers=filing.tickers,
            entities=[filing.company_name] + filing.tickers,
            collected_at=filing.collected_at,
            metadata={
                "filing_type": filing.filing_type,
                "company_name": filing.company_name,
                "cik": filing.cik,
                "items": filing.items,
            },
        )

    def collect_yahoo_finance(
        self,
        tickers: List[str],
        max_per_ticker: int = 10,
    ) -> List[UnifiedArticle]:
        """
        Collect articles from Yahoo Finance.

        Args:
            tickers: List of stock tickers
            max_per_ticker: Maximum articles per ticker

        Returns:
            List of UnifiedArticle objects
        """
        logger.info(f"Collecting Yahoo Finance articles for {len(tickers)} tickers...")

        articles = self.yahoo_collector.collect(
            tickers=tickers,
            max_per_ticker=max_per_ticker,
        )

        return [self._convert_yahoo_article(a) for a in articles]

    def collect_newsapi(
        self,
        companies: List[str],
        days_back: int = 7,
        articles_per_company: int = 20,
    ) -> List[UnifiedArticle]:
        """
        Collect articles from NewsAPI (Reuters, etc.).

        Args:
            companies: List of company names or tickers
            days_back: How many days back to search
            articles_per_company: Max articles per company

        Returns:
            List of UnifiedArticle objects
        """
        logger.info(f"Collecting NewsAPI articles for {len(companies)} companies...")

        articles = self.newsapi_collector.collect_by_companies(
            companies=companies,
            days_back=days_back,
            articles_per_company=articles_per_company,
        )

        return [self._convert_newsapi_article(a) for a in articles]

    def collect_sec_edgar(
        self,
        tickers: List[str],
        filing_type: str = "8-K",
        filings_per_company: int = 10,
    ) -> List[UnifiedArticle]:
        """
        Collect 8-K filings from SEC EDGAR.

        Args:
            tickers: List of stock tickers
            filing_type: Filing type (default: "8-K")
            filings_per_company: Max filings per company

        Returns:
            List of UnifiedArticle objects
        """
        logger.info(f"Collecting SEC EDGAR {filing_type} filings for {len(tickers)} tickers...")

        filings = self.edgar_collector.collect(
            tickers=tickers,
            filing_type=filing_type,
            filings_per_company=filings_per_company,
        )

        return [self._convert_edgar_filing(f) for f in filings]

    def collect(
        self,
        tickers: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        yahoo_max_per_ticker: int = 10,
        newsapi_days_back: int = 7,
        newsapi_articles_per_company: int = 20,
        edgar_filings_per_company: int = 10,
    ) -> List[UnifiedArticle]:
        """
        Collect articles from all specified sources.

        Args:
            tickers: List of stock tickers. If None, uses default top stocks.
            sources: List of sources to use. Options:
                    ["yahoo_finance", "newsapi", "sec_edgar"]
                    If None, uses all sources.
            yahoo_max_per_ticker: Max articles per ticker from Yahoo Finance
            newsapi_days_back: Days back for NewsAPI search
            newsapi_articles_per_company: Max articles per company from NewsAPI
            edgar_filings_per_company: Max filings per company from SEC EDGAR

        Returns:
            List of UnifiedArticle objects from all sources
        """
        # Default tickers
        if tickers is None:
            tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META",
                "NVDA", "TSLA", "JPM", "V", "JNJ",
            ]

        # Default sources
        if sources is None:
            sources = ["yahoo_finance", "sec_edgar"]
            if self.newsapi_key:
                sources.append("newsapi")

        all_articles = []
        seen_urls = set()

        # Collect from each source
        if "yahoo_finance" in sources:
            try:
                articles = self.collect_yahoo_finance(
                    tickers=tickers,
                    max_per_ticker=yahoo_max_per_ticker,
                )
                for article in articles:
                    if article.url not in seen_urls:
                        seen_urls.add(article.url)
                        all_articles.append(article)
                logger.info(f"Yahoo Finance: {len(articles)} articles")
            except Exception as e:
                logger.error(f"Yahoo Finance collection failed: {e}")

        if "newsapi" in sources:
            try:
                articles = self.collect_newsapi(
                    companies=tickers,  # Use tickers as company names
                    days_back=newsapi_days_back,
                    articles_per_company=newsapi_articles_per_company,
                )
                for article in articles:
                    if article.url not in seen_urls:
                        seen_urls.add(article.url)
                        all_articles.append(article)
                logger.info(f"NewsAPI: {len(articles)} articles")
            except Exception as e:
                logger.error(f"NewsAPI collection failed: {e}")

        if "sec_edgar" in sources:
            try:
                articles = self.collect_sec_edgar(
                    tickers=tickers,
                    filings_per_company=edgar_filings_per_company,
                )
                for article in articles:
                    if article.url not in seen_urls:
                        seen_urls.add(article.url)
                        all_articles.append(article)
                logger.info(f"SEC EDGAR: {len(articles)} filings")
            except Exception as e:
                logger.error(f"SEC EDGAR collection failed: {e}")

        logger.info(f"Total collected: {len(all_articles)} unique articles")
        return all_articles

    def save(
        self,
        articles: List[UnifiedArticle],
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

    def load(self, input_path: str) -> List[UnifiedArticle]:
        """
        Load articles from file.

        Args:
            input_path: Input file path

        Returns:
            List of UnifiedArticle objects
        """
        input_path = Path(input_path)
        articles = []

        if input_path.suffix == ".jsonl" or not input_path.suffix:
            with open(input_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        articles.append(UnifiedArticle(**data))
        else:
            with open(input_path) as f:
                data = json.load(f)
                for item in data:
                    articles.append(UnifiedArticle(**item))

        logger.info(f"Loaded {len(articles)} articles from {input_path}")
        return articles


def main():
    """Example usage of the data collection pipeline."""
    import argparse

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Collect financial news articles")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL"],
        help="Stock tickers to collect news for",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["yahoo_finance", "sec_edgar"],
        choices=["yahoo_finance", "newsapi", "sec_edgar"],
        help="News sources to use",
    )
    parser.add_argument(
        "--newsapi-key",
        help="NewsAPI key (or set NEWSAPI_KEY env var)",
    )
    parser.add_argument(
        "--output",
        default="data/collected_articles.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        default="jsonl",
        choices=["jsonl", "json"],
        help="Output format",
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = DataCollectionPipeline(newsapi_key=args.newsapi_key)

    # Collect articles
    articles = pipeline.collect(
        tickers=args.tickers,
        sources=args.sources,
    )

    # Save results
    pipeline.save(articles, args.output, args.format)

    print(f"\nCollected {len(articles)} articles")
    print(f"Saved to: {args.output}")

    # Print sample
    if articles:
        print("\nSample article:")
        sample = articles[0]
        print(f"  Title: {sample.title[:80]}...")
        print(f"  Source: {sample.source} ({sample.source_type})")
        print(f"  Tickers: {sample.tickers}")
        print(f"  Text length: {len(sample.text)} chars")


if __name__ == "__main__":
    main()
