# Agent Labeling Instructions (v2.1 Simplified)

You will label financial news articles with entity-level sentiment.

## Output Format

For EACH article, output one JSON line (NO markdown, NO explanation):

```json
{
  "id": "article_id",
  "entities": [
    {
      "canonical_id": "AAPL",
      "canonical_name": "Apple Inc.",
      "type": "ORG",
      "linked_ticker": "AAPL",
      "linked_company": null,
      "ner_mention_texts": ["Apple Inc.", "Apple"],
      "coref_mention_texts": ["the company", "it"],
      "sentiment_score": 0.5
    }
  ]
}
```

**CRITICAL**: Do NOT include `start_char`/`end_char` positions. Only text strings.
**CRITICAL**: Do NOT include the article `text` in output.

---

## Entity Types - EXTRACT ALL SIX TYPES

### ORG (Companies & Organizations)
- Company names: "Apple Inc.", "Goldman Sachs", "Tesla"
- Institutions: "Federal Reserve", "SEC", "NYSE", "EU", "OPEC"
- Extract ALL name variations: "Cipher Mining Inc." AND "Cipher Mining"

### TICKER (Stock Symbols)
- Trading symbols: "AAPL", "TSLA", "GS", "CIFR"
- From parentheses: "(NYSE:GS)" → extract "GS"
- NOT tickers: NYSE, SEC, FBI, CEO, CFO, ETF, IPO (these are acronyms)

### PERSON (Individual Names)
- "Tim Cook", "Elon Musk", "Jerome Powell"
- Both full name AND last-name-only mentions: "Cook"

### MONEY (ALL Monetary Amounts) - IMPORTANT, DO NOT SKIP
- Dollar amounts: "$1.5 billion", "$500 million", "$14.76"
- Other currencies: "€200 million", "¥50 billion"
- Stock prices: "$142.50"
- Revenue/earnings: "$2.4 billion in revenue"
- **Extract EVERY monetary amount in the article**

### PERCENT (ALL Percentage Values) - IMPORTANT, DO NOT SKIP
- Growth rates: "15%", "+2.5%", "-1.17%"
- Market changes: "rose 3.2%", "fell 0.5%"
- Financial ratios: "25.4% return"
- **Extract EVERY percentage in the article**

### DATE (ALL Dates & Time Periods) - IMPORTANT, DO NOT SKIP
- Quarters: "Q3 2024", "Q1 2025"
- Specific dates: "January 15, 2025", "December 2024"
- Fiscal periods: "fiscal year 2025", "full year 2024"
- Relative dates: "latest trading day", "last quarter"
- Years: "2024", "2025" (when referring to fiscal/calendar year)
- **Extract EVERY date reference in the article**

---

## NOT Entities - DO NOT Label

- Concepts: "AI", "artificial intelligence", "blockchain"
- Generic terms: "wealth manager", "investors", "analysts"
- Technologies: "semiconductors", "NAND flash"
- Locations: "Cleveland", "Silicon Valley" (unless government actor)
- Regulations: "Dodd-Frank", "ESG rules"
- Rankings: "Zacks Rank", "Strong Buy", "Outperform"

---

## Linking Rules

### Ticker-Company Pairs
When BOTH company name AND ticker appear:
- Create SEPARATE ORG and TICKER entities
- Use SAME `canonical_id` (use the ticker symbol)
- ORG has `linked_ticker`, TICKER has `linked_company`
- Both have the SAME `sentiment_score`

### Same Entity, Different Names
"Con Edison" and "Consolidated Edison, Inc." = SAME entity.
Use ONE entry with all name variants in `ner_mention_texts`:
```json
{
  "canonical_id": "ED",
  "canonical_name": "Consolidated Edison, Inc.",
  "type": "ORG",
  "ner_mention_texts": ["Consolidated Edison, Inc.", "Con Edison"],
  ...
}
```

---

## Sentiment Scoring

Score -1.0 to +1.0 for **ORG, TICKER, PERSON only**. Use `null` for MONEY, PERCENT, DATE.

| Score | Meaning | Example |
|-------|---------|---------|
| -1.0 to -0.7 | Very negative | Bankruptcy, fraud |
| -0.4 to -0.6 | Moderately negative | Earnings miss, downgrade |
| -0.1 to -0.3 | Slightly negative | Cautious outlook |
| 0.0 | Neutral | Factual mention |
| +0.1 to +0.3 | Slightly positive | Minor beat |
| +0.4 to +0.6 | Moderately positive | Earnings beat, upgrade |
| +0.7 to +1.0 | Very positive | Record performance |

**Read the CONTEXT** to determine sentiment. "increases despite market slip" = +0.2 to +0.3.

---

## Coreference Mentions

`coref_mention_texts`: pronouns and generic references to entities:
- "it", "they", "the company", "the firm", "the tech giant"
- "he", "she", "the CEO", "the founder"
- Only include when the reference clearly refers to a specific entity

---

## Extraction Checklist

Before finishing each article, verify:
- [ ] ALL company/org names extracted (ORG)
- [ ] ALL ticker symbols extracted (TICKER)
- [ ] ALL person names extracted (PERSON)
- [ ] ALL dollar/currency amounts extracted (MONEY)
- [ ] ALL percentages extracted (PERCENT)
- [ ] ALL dates/periods extracted (DATE)
- [ ] Ticker-company pairs share same canonical_id
- [ ] Sentiment reflects article tone (not all neutral)
- [ ] Same entity with different names grouped under one entry
