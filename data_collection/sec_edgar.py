"""SEC EDGAR 8-K Filings Collector.

Collects 8-K filings (current reports) from SEC EDGAR.
8-K filings contain material corporate events: earnings, M&A, leadership changes, etc.

No API key required - SEC EDGAR is fully open.
"""

import time
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


logger = logging.getLogger(__name__)


@dataclass
class Filing:
    """Represents an SEC filing."""
    title: str
    text: str
    url: str
    source: str
    published_date: str
    tickers: List[str]
    collected_at: str
    filing_type: str
    company_name: str
    cik: str
    items: List[str]  # 8-K item numbers (e.g., "Item 2.02")

    def to_dict(self) -> dict:
        return asdict(self)


# Map CIK to ticker for top companies
# You can expand this or load from a file
CIK_TO_TICKER = {
    "320193": "AAPL",
    "789019": "MSFT",
    "1652044": "GOOGL",
    "1018724": "AMZN",
    "1326801": "META",
    "1045810": "NVDA",
    "1318605": "TSLA",
    "19617": "JPM",
    "886982": "GS",
    "70858": "BAC",
    "200406": "JNJ",
    "21344": "KO",
    "66740": "MMM",
    "51143": "IBM",
    "1403161": "V",
    "1141391": "MA",
    "1288776": "GOOG",
    "1467373": "CRM",
    "2488": "AMD",
    "50863": "INTC",
}

# Ticker to CIK (reverse mapping)
TICKER_TO_CIK = {v: k for k, v in CIK_TO_TICKER.items()}


class SECEdgarCollector:
    """
    Collect 8-K filings from SEC EDGAR.

    8-K filings are "current reports" that companies must file when
    material events occur, including:
    - Item 1.01: Entry into Material Agreement
    - Item 2.01: Completion of Acquisition
    - Item 2.02: Results of Operations (Earnings)
    - Item 2.05: Costs from Exit Activities
    - Item 5.02: Departure/Election of Directors/Officers
    - Item 7.01: Regulation FD Disclosure
    - Item 8.01: Other Events

    Attributes:
        base_url: SEC EDGAR API endpoint
        delay: Delay between requests
    """

    # SEC EDGAR API endpoints
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
    FULL_TEXT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession_nodash}.txt"

    # Required headers per SEC policy
    HEADERS = {
        "User-Agent": "PersonalResearch contact@example.com",  # Update with your email
        "Accept-Encoding": "gzip, deflate",
    }

    def __init__(
        self,
        user_agent: Optional[str] = None,
        delay: float = 0.1,  # SEC allows 10 requests/second
    ):
        """
        Initialize SEC EDGAR collector.

        Args:
            user_agent: User agent string (SEC requires identification)
            delay: Delay between requests (SEC allows 10 req/sec)
        """
        if requests is None:
            raise ImportError("requests is required. Install with: pip install requests")

        if user_agent:
            self.HEADERS["User-Agent"] = user_agent

        self.delay = delay

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make a request to SEC EDGAR with rate limiting."""
        try:
            time.sleep(self.delay)
            response = requests.get(url, headers=self.HEADERS, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def get_company_filings(
        self,
        cik: str,
        filing_type: str = "8-K",
        limit: int = 50,
    ) -> List[Dict]:
        """
        Get recent filings for a company.

        Args:
            cik: Company CIK number (can include leading zeros)
            filing_type: Filing type to filter (e.g., "8-K", "10-K")
            limit: Maximum filings to return

        Returns:
            List of filing metadata dicts
        """
        # Pad CIK to 10 digits
        cik = cik.zfill(10)

        url = self.SUBMISSIONS_URL.format(cik=cik)
        response = self._make_request(url)

        if not response:
            return []

        try:
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            return []

        company_name = data.get("name", "")
        recent = data.get("filings", {}).get("recent", {})

        if not recent:
            return []

        filings = []
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form == filing_type and len(filings) < limit:
                filings.append({
                    "form": form,
                    "filing_date": dates[i] if i < len(dates) else "",
                    "accession": accessions[i] if i < len(accessions) else "",
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                    "company_name": company_name,
                    "cik": cik.lstrip("0"),
                })

        return filings

    def _extract_8k_items(self, text: str) -> List[str]:
        """Extract 8-K item numbers from filing text."""
        items = []
        # Pattern: Item X.XX
        pattern = r"Item\s+(\d+\.\d+)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:
            item = f"Item {match}"
            if item not in items:
                items.append(item)

        return items

    def _clean_filing_text(self, html: str) -> str:
        """Clean HTML filing to plain text."""
        if BeautifulSoup is None:
            # Basic HTML tag removal without BeautifulSoup
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        soup = BeautifulSoup(html, "html.parser")

        # Remove scripts, styles
        for element in soup(["script", "style"]):
            element.decompose()

        text = soup.get_text(separator=" ")

        # Clean whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def get_filing_text(
        self,
        cik: str,
        accession: str,
        primary_doc: str,
    ) -> Optional[str]:
        """
        Get the text content of a filing.

        Args:
            cik: Company CIK
            accession: Accession number
            primary_doc: Primary document filename

        Returns:
            Filing text content
        """
        # Remove dashes from accession for URL
        accession_nodash = accession.replace("-", "")
        cik = cik.lstrip("0")

        url = self.FILING_URL.format(
            cik=cik,
            accession=accession_nodash,
            filename=primary_doc,
        )

        response = self._make_request(url)
        if not response:
            return None

        return self._clean_filing_text(response.text)

    def collect_for_company(
        self,
        ticker_or_cik: str,
        filing_type: str = "8-K",
        limit: int = 20,
        fetch_text: bool = True,
    ) -> List[Filing]:
        """
        Collect filings for a single company.

        Args:
            ticker_or_cik: Stock ticker or CIK number
            filing_type: Filing type to collect
            limit: Maximum filings
            fetch_text: Whether to fetch full filing text

        Returns:
            List of Filing objects
        """
        # Convert ticker to CIK if needed
        if ticker_or_cik.upper() in TICKER_TO_CIK:
            cik = TICKER_TO_CIK[ticker_or_cik.upper()]
            ticker = ticker_or_cik.upper()
        else:
            cik = ticker_or_cik
            ticker = CIK_TO_TICKER.get(cik.lstrip("0"), "")

        logger.info(f"Collecting {filing_type} filings for {ticker or cik}...")

        filing_metadata = self.get_company_filings(cik, filing_type, limit)
        filings = []

        for meta in filing_metadata:
            if fetch_text:
                text = self.get_filing_text(
                    meta["cik"],
                    meta["accession"],
                    meta["primary_doc"],
                )
            else:
                text = ""

            # Extract 8-K items
            items = self._extract_8k_items(text) if text else []

            # Create title from items
            if items:
                items_str = ", ".join(items[:3])
                title = f"{meta['company_name']} {filing_type}: {items_str}"
            else:
                title = f"{meta['company_name']} {filing_type} Filing"

            filing = Filing(
                title=title,
                text=text or f"{filing_type} filing for {meta['company_name']}",
                url=self.FILING_URL.format(
                    cik=meta["cik"],
                    accession=meta["accession"].replace("-", ""),
                    filename=meta["primary_doc"],
                ),
                source="SEC EDGAR",
                published_date=meta["filing_date"],
                tickers=[ticker] if ticker else [],
                collected_at=datetime.now().isoformat(),
                filing_type=filing_type,
                company_name=meta["company_name"],
                cik=meta["cik"],
                items=items,
            )
            filings.append(filing)

        logger.info(f"Collected {len(filings)} {filing_type} filings")
        return filings

    def collect(
        self,
        tickers: Optional[List[str]] = None,
        filing_type: str = "8-K",
        filings_per_company: int = 10,
        fetch_text: bool = True,
    ) -> List[Filing]:
        """
        Collect filings for multiple companies.

        Args:
            tickers: List of stock tickers. If None, uses default list.
            filing_type: Filing type to collect
            filings_per_company: Max filings per company
            fetch_text: Whether to fetch full filing text

        Returns:
            List of Filing objects
        """
        if tickers is None:
            tickers = list(TICKER_TO_CIK.keys())[:20]

        all_filings = []

        for ticker in tickers:
            if ticker.upper() not in TICKER_TO_CIK:
                logger.warning(f"No CIK mapping for {ticker}, skipping")
                continue

            filings = self.collect_for_company(
                ticker,
                filing_type=filing_type,
                limit=filings_per_company,
                fetch_text=fetch_text,
            )
            all_filings.extend(filings)

        logger.info(f"Collected {len(all_filings)} total filings")
        return all_filings

    def save(
        self,
        filings: List[Filing],
        output_path: str,
        format: str = "jsonl",
    ) -> None:
        """
        Save collected filings to file.

        Args:
            filings: List of filings to save
            output_path: Output file path
            format: Output format ("jsonl" or "json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for filing in filings:
                    f.write(json.dumps(filing.to_dict()) + "\n")
        else:
            with open(output_path, "w") as f:
                json.dump([f.to_dict() for f in filings], f, indent=2)

        logger.info(f"Saved {len(filings)} filings to {output_path}")


def add_ticker_mapping(ticker: str, cik: str) -> None:
    """
    Add a ticker to CIK mapping.

    You can find CIK numbers at: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany

    Args:
        ticker: Stock ticker symbol
        cik: SEC CIK number
    """
    CIK_TO_TICKER[cik.lstrip("0")] = ticker.upper()
    TICKER_TO_CIK[ticker.upper()] = cik.lstrip("0")
