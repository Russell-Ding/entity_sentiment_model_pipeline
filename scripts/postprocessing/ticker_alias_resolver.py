#!/usr/bin/env python3
"""Resolve detected entity names (e.g. "Apple Inc.") to stock tickers (e.g. AAPL).

Inputs:
  - data/raw/sp500_components_20260517.csv  (seed: ticker, company name)
  - data/reference/ticker_aliases_extras.csv  (manual: famous non-SP500 aliases)
  - data/reference/non_company_blocklist.csv  (governments, news outlets, generic words)

Algorithm (first match wins):
  1. Lowercase + strip corporate suffix loop ("Apple Inc." -> "apple").
  2. Check blocklist -> return None.
  3. Check exact-normalized lookup -> return ticker.
  4. Check raw-uppercase as ticker (entity_type == 'TICKER' or short all-caps).
  5. Return None.

Usage as a module:
    from ticker_alias_resolver import TickerAliasResolver
    resolver = TickerAliasResolver()
    result = resolver.resolve("Apple Inc.", "ORG")
    # -> {"ticker": "AAPL", "company": "Apple Inc.", "source": "sp500"}
"""

from __future__ import annotations

import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SP500 = PROJECT_ROOT / "data" / "raw" / "sp500_components_20260517.csv"
DEFAULT_EXTRAS = PROJECT_ROOT / "data" / "reference" / "ticker_aliases_extras.csv"
DEFAULT_BLOCKLIST = PROJECT_ROOT / "data" / "reference" / "non_company_blocklist.csv"

# Strip in a loop: catches "Apple Inc., Ltd." -> "Apple Inc." -> "Apple"
_SUFFIX_RE = re.compile(
    r"(?:\s+|,\s*)(?:"
    r"inc|incorporated|corp|corporation|"
    r"co|company|companies|"
    r"ltd|limited|llc|plc|"
    r"holdings?|holding|group|grp|"
    r"sa|nv|ag|se|kk|kg|gmbh|spa|s\.a\.|n\.v\.|a\.g\.|"
    r"class\s+[a-c]"
    r")\.?$",
    re.IGNORECASE,
)
# Strip dangling stop-words ("&", "and", "the") left over after suffix removal
_TRAIL_STOP_RE = re.compile(r"\s+(?:&|and|the|of)\s*$", re.IGNORECASE)
# Domain-style suffixes in company names ("Amazon.com Inc." -> "Amazon Inc.")
_DOMAIN_RE = re.compile(r"\.(?:com|net|org|io|ai)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

# A short all-caps token that looks like a ticker (1-5 letters, optional .XX or -XX class suffix)
_TICKER_SHAPE_RE = re.compile(r"^\$?([A-Z]{1,5}(?:[-.][A-Z]{1,3})?)$")


def normalize(text: str) -> str:
    """Aggressive lowercase + suffix-strip normalization for alias matching.

    "Apple Inc." -> "apple"
    "Apple, Inc"  -> "apple"
    "JPMorgan Chase & Co." -> "jpmorgan chase"
    "Berkshire Hathaway Inc. Class B" -> "berkshire hathaway class b"
    """
    s = (text or "").strip()
    if not s:
        return ""
    # Strip wrapping quotes/punctuation
    s = s.strip(".,;:'\"`")
    s = s.lower()
    # Strip domain suffix early ("amazon.com" -> "amazon")
    s = _DOMAIN_RE.sub("", s)
    # Iterative suffix + trailing-stop-word strip
    prev = None
    while s != prev:
        prev = s
        s = _SUFFIX_RE.sub("", s).strip()
        s = s.strip(".,;:&")
        s = _TRAIL_STOP_RE.sub("", s).strip()
        s = s.strip(".,;:&")
    # Drop remaining punctuation -> spaces
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


@dataclass(frozen=True)
class ResolveResult:
    ticker: str
    company: str
    source: str  # "sp500", "extras", "ticker_shape"


class TickerAliasResolver:
    def __init__(
        self,
        sp500_csv: Path = DEFAULT_SP500,
        extras_csv: Optional[Path] = DEFAULT_EXTRAS,
        blocklist_csv: Optional[Path] = DEFAULT_BLOCKLIST,
    ) -> None:
        self.alias_map: Dict[str, Tuple[str, str, str]] = {}  # norm_alias -> (ticker, company, source)
        self.ticker_set: Set[str] = set()  # all valid tickers (for ticker-shape match)
        self.blocklist: Set[str] = set()   # normalized names that should never resolve
        self.ticker_canonical: Dict[str, str] = {}  # raw ticker -> preferred form (e.g. BRK.B -> BRK-B)

        self._load_blocklist(blocklist_csv)
        self._load_sp500(sp500_csv)
        if extras_csv is not None:
            self._load_extras(extras_csv)

    def _load_blocklist(self, path: Optional[Path]) -> None:
        if path is None or not path.exists():
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "").strip()
                if name:
                    self.blocklist.add(normalize(name))

    def _load_sp500(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"SP500 CSV not found: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("ticker", "").strip()
                company = row.get("company", "").strip()
                eodhd_sym = (row.get("eodhd_symbol") or "").strip()
                if not ticker or not company:
                    continue
                # Prefer EODHD ticker form when it differs (e.g. BRK.B -> BRK-B)
                canonical = eodhd_sym.split(".")[0] if eodhd_sym else ticker
                if canonical and canonical != ticker:
                    self.ticker_canonical[ticker] = canonical
                self.ticker_set.add(ticker)
                self.ticker_set.add(canonical)
                # Map the full company name
                norm_full = normalize(company)
                if norm_full and norm_full not in self.alias_map:
                    self.alias_map[norm_full] = (canonical, company, "sp500")
                # Also map the ticker itself (so "AAPL" -> AAPL works)
                self.alias_map[ticker.lower()] = (canonical, company, "sp500")
                self.alias_map[canonical.lower()] = (canonical, company, "sp500")
                # First word as a short alias (e.g. "Apple" from "Apple Inc.") ONLY when
                # it's unambiguous — added in second pass below.

        # Second pass: 2-word prefixes as short aliases, only when unambiguous.
        # 1-word prefixes are intentionally NOT generated — common English words
        # like "Federal", "American", "General" would otherwise resolve to the
        # one SP500 company that happens to start with them (e.g. "Federal" -> FRT).
        short_alias_counts: Dict[str, int] = {}
        short_alias_to_ticker: Dict[str, Tuple[str, str]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("ticker", "").strip()
                company = row.get("company", "").strip()
                if not ticker or not company:
                    continue
                norm_full = normalize(company)
                words = norm_full.split()
                if len(words) > 2:
                    short = " ".join(words[:2])
                    if short and short not in self.alias_map:
                        short_alias_counts[short] = short_alias_counts.get(short, 0) + 1
                        short_alias_to_ticker[short] = (ticker, company)
        for short, count in short_alias_counts.items():
            if count == 1 and short not in self.blocklist:
                ticker, company = short_alias_to_ticker[short]
                self.alias_map[short] = (ticker, company, "sp500_short")

    def _load_extras(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                alias = (row.get("alias") or "").strip()
                ticker = (row.get("ticker") or "").strip()
                company = (row.get("company") or "").strip()
                if not alias:
                    continue
                norm = normalize(alias)
                if not norm:
                    continue
                # Empty ticker = explicitly known-no-ticker (private, foreign-only) -> still mark
                # so we don't fall through to ticker-shape match.
                if norm in self.alias_map:
                    old_ticker, old_company, old_source = self.alias_map[norm]
                    if old_ticker != (ticker or ""):
                        logging.warning(
                            "Extras alias '%s' (%s) overwriting existing %s mapping (%s -> %s was %s)",
                            alias, ticker or "(none)", old_source, norm, ticker or "(none)", old_ticker,
                        )
                self.alias_map[norm] = (ticker or "", company, "extras")
                if ticker:
                    self.ticker_set.add(ticker)

    def resolve(self, entity_text: str, entity_type: str = "ORG") -> Optional[ResolveResult]:
        """Resolve an entity text to a ticker. Returns None if unresolvable.

        entity_type is informational only ("TICKER" gives ticker-shape fallback priority).
        """
        if not entity_text:
            return None
        norm = normalize(entity_text)
        if not norm:
            return None
        # Blocklist first
        if norm in self.blocklist:
            return None
        # Direct alias match
        if norm in self.alias_map:
            ticker, company, source = self.alias_map[norm]
            if not ticker:  # known-no-ticker (e.g. OpenAI)
                return None
            return ResolveResult(ticker=ticker, company=company, source=source)
        # Ticker-shape match (e.g. raw "AAPL" not in name index)
        m = _TICKER_SHAPE_RE.match(entity_text.strip())
        if m:
            t = m.group(1)
            if t in self.ticker_set:
                canonical = self.ticker_canonical.get(t, t)
                # Look up the company for this ticker
                for k, (tk, comp, src) in self.alias_map.items():
                    if tk == canonical:
                        return ResolveResult(ticker=canonical, company=comp, source="ticker_shape")
                return ResolveResult(ticker=canonical, company="", source="ticker_shape")
        return None


# ---------------------------------------------------------------------------

def _summarize(resolver: TickerAliasResolver) -> None:
    print(f"Loaded {len(resolver.ticker_set):,} tickers, "
          f"{len(resolver.alias_map):,} alias entries, "
          f"{len(resolver.blocklist):,} blocklist entries.")
    examples = [
        ("Apple", "ORG"), ("Apple Inc.", "ORG"), ("Apple Inc", "ORG"),
        ("AAPL", "TICKER"),
        ("Alphabet", "ORG"), ("Alphabet Inc.", "ORG"), ("Google", "ORG"),
        ("Meta", "ORG"), ("Facebook", "ORG"),
        ("Berkshire Hathaway", "ORG"), ("Berkshire", "ORG"),
        ("TSMC", "ORG"), ("Taiwan Semiconductor Manufacturing Co.", "ORG"),
        ("Sony", "ORG"), ("Sony Corp.", "ORG"),
        ("JPMorgan", "ORG"), ("JP Morgan", "ORG"), ("JPMorgan Chase & Co.", "ORG"),
        ("OpenAI", "ORG"), ("Foxconn", "ORG"), ("Huawei", "ORG"),
        ("Fed", "ORG"), ("Federal Reserve", "ORG"), ("Reuters", "ORG"),
        ("Group", "ORG"), ("Inc", "ORG"),
        ("Donald", "PERSON"), ("Tim Cook", "PERSON"),
    ]
    print()
    print(f"{'ENTITY':<42} {'TYPE':<8} {'-> TICKER':<10} {'COMPANY':<35} SOURCE")
    print("-" * 110)
    for text, etype in examples:
        r = resolver.resolve(text, etype)
        if r:
            print(f"{text:<42} {etype:<8} {r.ticker:<10} {r.company[:33]:<35} {r.source}")
        else:
            print(f"{text:<42} {etype:<8} {'-':<10} {'(unresolved)':<35} ")


def main() -> None:
    resolver = TickerAliasResolver()
    _summarize(resolver)


if __name__ == "__main__":
    main()
