"""Data collection module for financial news articles."""

try:
    from .yahoo_finance import YahooFinanceCollector
except (ImportError, TypeError):
    YahooFinanceCollector = None

from .newsapi_collector import NewsAPICollector
from .sec_edgar import SECEdgarCollector
from .pipeline import DataCollectionPipeline

__all__ = [
    "YahooFinanceCollector",
    "NewsAPICollector",
    "SECEdgarCollector",
    "DataCollectionPipeline",
]
