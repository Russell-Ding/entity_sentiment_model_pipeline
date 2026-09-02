# Entity & Sentiment Labeling Specification v2

**Version**: 2.0
**Date**: 2026-01-22
**Purpose**: Direct Claude labeling without Python code extraction

---

## CRITICAL: Direct Labeling Approach

**DO NOT** use Python code, regex, or keyword matching for entity extraction.
**DO** use Claude's language understanding to directly identify entities and sentiment.

The labeler must READ the article and IDENTIFY entities using comprehension, not pattern matching.

---

## CRITICAL: Haiku 4.5 Agent Labeling Rules

### Why We Use LLM Agents Instead of Python Code

**The whole point of using Haiku 4.5 for labeling is that LLMs understand language in ways Python code cannot:**

| Capability | Python/Regex | Haiku 4.5 Agent |
|------------|--------------|-----------------|
| Pattern matching | Yes | Yes |
| Context understanding | No | **Yes** |
| Coreference resolution | Limited | **Yes** |
| Implied entity detection | No | **Yes** |
| Sentiment from context | No | **Yes** |
| Sarcasm/irony detection | No | **Yes** |
| Domain knowledge | No | **Yes** |

**If we wanted pattern matching, we wouldn't need an LLM - we'd just write regex!**

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                LABELING PIPELINE                            │
│                                                             │
│  Python Orchestration          Haiku 4.5 Agents            │
│  (file I/O, API calls)         (ALL intelligent work)      │
│                                                             │
│  ┌──────────────┐              ┌──────────────────────┐    │
│  │ Read JSONL   │───Article───►│  LABELER AGENT       │    │
│  │ file         │              │  - Reads & understands│    │
│  └──────────────┘              │  - Identifies entities│    │
│                                │  - Outputs JSON       │    │
│                                └──────────┬───────────┘    │
│                                           │                 │
│                                           ▼                 │
│                                ┌──────────────────────┐    │
│                                │  VALIDATOR AGENT     │    │
│                                │  - Checks correctness│    │
│                                │  - Provides feedback │    │
│                                └──────────┬───────────┘    │
│                                           │                 │
│  ┌──────────────┐              ◄──────────┘                │
│  │ Write JSONL  │                                          │
│  │ output       │              (feedback loop if needed)   │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

### Rules for Haiku 4.5 Agents

1. **NO PYTHON SCRIPTS for NER/Sentiment**: The agent must NOT write or execute Python code to extract entities or calculate sentiment. All extraction must come directly from the LLM's language understanding.

2. **Direct JSON Output**: The agent should directly output the labeled JSON structure after reading and understanding the article text.

3. **Why This Matters**:
   - Python-based extraction uses pattern matching, which misses context
   - LLM comprehension understands coreference, implied entities, and nuanced sentiment
   - Code-based approaches produce lower quality labels that don't leverage the model's capabilities
   - **If the agent writes Python code, it defeats the entire purpose of using an LLM!**

4. **Correct Workflow**:
   ```
   Input Article → Haiku READS & UNDERSTANDS → Direct JSON labels output
   ```

   **WRONG (defeats the purpose)**:
   ```
   Input Article → Haiku writes Python → Python extracts entities → JSON output
   ```

5. **Prompt Template Must Include**:
   ```
   You are a LABELING AGENT, not a programmer.
   READ the article and USE YOUR LANGUAGE UNDERSTANDING to identify entities.
   DO NOT write any Python code, regex, or programmatic extraction.
   Output your analysis directly as JSON.
   If you write code, your response will be rejected.
   ```

### What Python Code IS Allowed For

The ONLY Python code in the pipeline is for:
- Reading input JSONL files
- Calling the Anthropic API
- Parsing JSON responses from Haiku
- **Validating character positions** (simple `text[start:end] == mention` check)
- Writing output JSONL files

This is orchestration, not intelligence. ALL the intelligent labeling work is done by Haiku.

---

## CRITICAL: Agent Output Simplification (Version 2.1)

### Problem with Previous Approach

We discovered that when agents generate both entity labels AND full article body text AND character positions, errors accumulate:

1. **Text Truncation**: Agents sometimes truncate article body to just headlines
2. **Position Errors**: Character positions calculated on full text but saved with truncated text
3. **Data Duplication**: Article body duplicated across source and labeled files

### New Simplified Workflow

**Agents ONLY generate semantic understanding** (the LLM's strength):
- Entity canonical IDs
- Entity types
- Entity mention **text strings** (NOT positions)
- Sentiment scores
- Entity relationships (linked_ticker, linked_company)

**Python post-processing handles mechanical tasks** (no LLM needed):
- Map full article body from source files using article ID
- Calculate character positions by searching for mention text in article
- Validate positions: `text[start:end] == mention_text`
- Handle edge cases (multiple occurrences, closest match)

### Agent Output Schema (Simplified)

Agents should output this simplified structure:

```json
{
  "id": "article_123",
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
    },
    {
      "canonical_id": "AAPL",
      "canonical_name": "AAPL",
      "type": "TICKER",
      "linked_ticker": null,
      "linked_company": "Apple Inc.",
      "ner_mention_texts": ["AAPL"],
      "coref_mention_texts": [],
      "sentiment_score": 0.5
    }
  ]
}
```

**Note**: NO `text` field, NO `start_char`/`end_char` in mentions!

### Python Post-Processing Steps

After agent labeling, Python script performs:

1. **Body Text Mapping**:
   ```python
   # Load source article by ID
   source_article = load_source_article(labeled_article["id"])
   labeled_article["text"] = source_article["text"]
   ```

2. **Position Calculation**:
   ```python
   def find_all_positions(text, substring):
       positions = []
       start = 0
       while True:
           pos = text.find(substring, start)
           if pos == -1:
               break
           positions.append((pos, pos + len(substring)))
           start = pos + 1
       return positions

   # For each mention text, find positions
   for mention_text in entity["ner_mention_texts"]:
       positions = find_all_positions(article_text, mention_text)
       # Convert to structured mentions with positions
       for start, end in positions:
           ner_mentions.append({
               "text": mention_text,
               "start_char": start,
               "end_char": end
           })
   ```

3. **Validation**:
   ```python
   # Verify all positions are correct
   for mention in entity["ner_mentions"]:
       assert text[mention["start_char"]:mention["end_char"]] == mention["text"]
   ```

### Benefits of This Approach

1. **No Text Truncation**: Body text comes directly from source, never processed by LLM
2. **No Position Errors**: Positions calculated mechanically, not by LLM guessing
3. **Faster Labeling**: Agents don't waste tokens copying body text
4. **Lower Cost**: Shorter agent outputs = fewer output tokens
5. **Better Separation**: LLM does understanding, Python does mechanics

### Agent Prompt Template (Updated)

When prompting labeling agents, include:

```
You are a LABELING AGENT that identifies entities and sentiment in financial news.

CRITICAL SIMPLIFICATION:
- DO NOT include the article "text" field in your output
- DO NOT calculate "start_char" or "end_char" positions
- ONLY output entity understanding: types, IDs, mention texts, sentiment

Your output should be:
{
  "id": "article_id",
  "entities": [
    {
      "canonical_id": "...",
      "type": "...",
      "ner_mention_texts": ["text1", "text2"],
      "sentiment_score": ...
    }
  ]
}

Python post-processing will map the full article body and calculate positions.
```

### Backward Compatibility

For existing labeled data with positions, this approach can be used to **fix incorrect positions**:

1. Extract mention texts from existing data
2. Re-map body text from source files
3. Recalculate positions using Python search
4. Validate and save corrected data

This is exactly how we fixed 8,697 position errors in our dataset.

---

## 1. Entity Types

### 1.1 TICKER (Stock Symbols)
**Definition**: Official stock market trading symbols

**Characteristics**:
- 1-5 uppercase letters
- Represents a company's stock on an exchange
- Often appears in parentheses: `(AAPL)`, `(NASDAQ:MSFT)`
- May have `$` prefix: `$TSLA`

**Examples**:
- `AAPL`, `MSFT`, `GOOGL`, `CIFR`, `NVDA`
- `(NYSE:GS)` → extract `GS`
- `$TSLA` → extract `TSLA`

**NOT Tickers** (these are ORG):
- `NYSE`, `SEC`, `ECB`, `FOMC`, `IMF`, `FED`, `CNBC`, `FBI`, `CIA`, `NATO`, `OPEC`, `FDA`, `CDC`, `WHO`, `UN`, `EU`, `BBC`, `CNN`, `CEO`, `CFO`, `COO`

**CRITICAL: Single-Letter Ticker Disambiguation**
Single-letter tickers require careful context analysis:
- `F` = Ford Motor Company - BUT only when clearly referring to Ford stock
- `F` in "ETF", "CFO", or other acronyms is NOT the Ford ticker
- Always verify the context before labeling single letters as tickers
- When in doubt, check if the surrounding text discusses the company (Ford, Ford Motor, automotive, etc.)

**Examples of F disambiguation**:
- "F stock rose 5%" → F is TICKER (Ford)
- "the ETF gained" → F is NOT a ticker (part of ETF acronym)
- "Ford (F) reported earnings" → F is TICKER (explicitly linked to Ford)
- "the CFO announced" → F is NOT a ticker (part of CFO title)

### 1.2 ORG (Organizations)
**Definition**: ALL organization names including companies and institutions

**CRITICAL**: When a company name appears, it MUST be labeled as ORG.

**Company Examples**:
- `Apple Inc.`, `Apple` (when referring to the company)
- `Cipher Mining Inc.`, `Cipher Mining`
- `Microsoft Corporation`, `Microsoft`
- `Goldman Sachs`, `JPMorgan Chase`
- `Tesla`, `Amazon`, `Meta`

**Institution Examples**:
- `Federal Reserve`, `the Fed`
- `Securities and Exchange Commission`, `SEC`
- `New York Stock Exchange`, `NYSE`
- `European Central Bank`, `ECB`

**Governmental/Supranational Organizations** (MUST be labeled as ORG):
- `EU`, `European Union` - supranational political/economic union
- `UN`, `United Nations`
- `NATO`, `North Atlantic Treaty Organization`
- `OPEC`, `Organization of the Petroleum Exporting Countries`
- `IMF`, `International Monetary Fund`
- `World Bank`
- `WTO`, `World Trade Organization`

**Note**: These are organizations that affect markets through policy/regulation. Always extract them as ORG when mentioned in financial news.

**Key Rule**: If the text refers to a company by name, label it as ORG. This includes:
- Full names: "Cipher Mining Inc."
- Short names: "Cipher Mining"
- Common names: "Apple" (when it's the tech company)

### 1.3 PERSON
**Definition**: Individual people's names

**Examples**:
- `Tim Cook`, `Elon Musk`, `Warren Buffett`
- `Jerome Powell`, `Janet Yellen`
- First mentions: `Tim Cook`
- Subsequent mentions: `Cook`

### 1.4 MONEY
**Definition**: Monetary amounts

**Examples**:
- `$1.5 billion`, `$500 million`
- `€200 million`, `¥50 billion`
- `$14.76` (stock price)

### 1.5 PERCENT
**Definition**: Percentage values

**Examples**:
- `15%`, `2.5%`, `+1.17%`
- `2.5 percentage points`

### 1.6 DATE
**Definition**: Dates and time periods

**Examples**:
- `Q3 2024`, `Q1 2025`
- `January 15, 2025`
- `fiscal year 2025`
- `latest trading day`

---

## 1.7 NOT an Entity (Do Not Label as ORG/TICKER)

These are commonly mistaken for entities but should NOT be labeled:

### Concepts & Abstract Terms
- `AI`, `artificial intelligence` (technology concept, not a company)
- `money laundering`, `fraud`, `insider trading` (crimes/activities)
- `tariff`, `sanctions`, `trade war` (policies/situations)
- `inflation`, `recession`, `bull market` (economic conditions)

### Generic Business Terms
- `wealth manager`, `financial advisor` (roles, not organizations)
- `alternative investments`, `private equity` (investment categories)
- `finance executives`, `tech workers` (generic people groups)
- `startup`, `fintech` (generic company types)

### Technologies & Products (unless company name)
- `NAND flash memory`, `semiconductors`, `chips` (product categories)
- `iPhone`, `Windows` (products - label the company instead: Apple, Microsoft)
- `ChatGPT`, `GPT-4` (products - label OpenAI if discussing the company)

### Regulations & Rules
- `ESG disclosure rules`, `climate regulations`
- `Dodd-Frank`, `Basel III` (regulations, not organizations)
- `antitrust law`, `securities law`

### Market Terms (OK for sentiment_expanded_mentions, NOT for ner_mentions)
- `US equity indexes`, `tech sector`, `financials`
- `S&P 500`, `Nasdaq`, `Dow Jones` (indices - see sentiment expansion)
- `large-cap stocks`, `growth stocks`

### Locations (NOT ORG)
- City names: `Cleveland`, `San Francisco`, `Garland, Texas`
- Countries: `China`, `Germany` (unless referring to government as actor)
- Regions: `Silicon Valley`, `Wall Street` (as location)

### Exception: When Locations ARE Organizations
- `China` as government actor: "China imposed tariffs" → ORG
- `The White House` as decision maker → ORG
- `Brussels` meaning EU leadership → use `EU` as ORG instead

---

## 2. Entity Extraction Rules

### 2.1 Complete Extraction
**Extract ALL mentions of each entity type**. Do not miss obvious entities.

**CRITICAL: Extract EVERY mention across the ENTIRE article**
- Read the ENTIRE article, not just the first sentence
- If "Monster Beverage" appears in sentence 1 AND sentence 2, extract BOTH mentions
- Company names often appear multiple times - capture ALL occurrences
- Do NOT stop after finding the first mention

**Common Mistake**: Only extracting entities from the headline/first sentence
**Correct Approach**: Scan the entire article text and extract every mention

**Example Article**:
```
Cipher Mining Inc. (CIFR) Increases Despite Market Slip: Here's What You Need to Know.
Cipher Mining Inc. (CIFR) reached $14.76 at the closing of the latest trading day,
reflecting a +1.17% change compared to its last close.
```

**Required Entities**:
| Entity | Type | Mentions |
|--------|------|----------|
| Cipher Mining Inc. | ORG | 2 mentions (positions 0-18 and 96-114) |
| CIFR | TICKER | 2 mentions (positions 20-24 and 116-120) |
| $14.76 | MONEY | 1 mention |
| +1.17% | PERCENT | 1 mention |
| latest trading day | DATE | 1 mention |

### 2.2 Ticker-Company Linking
When both ticker and company name appear, they should:
- Have the SAME `canonical_id` (use the ticker symbol)
- Be SEPARATE entities (different types)
- Share the SAME `sentiment_score`

**Example**:
```json
{
  "canonical_id": "CIFR",
  "canonical_name": "Cipher Mining Inc.",
  "type": "ORG",
  "linked_ticker": "CIFR",
  "ner_mentions": [
    {"text": "Cipher Mining Inc.", "start_char": 0, "end_char": 18},
    {"text": "Cipher Mining Inc.", "start_char": 96, "end_char": 114}
  ],
  "sentiment_score": 0.3
},
{
  "canonical_id": "CIFR",
  "canonical_name": "CIFR",
  "type": "TICKER",
  "linked_company": "Cipher Mining Inc.",
  "ner_mentions": [
    {"text": "CIFR", "start_char": 20, "end_char": 24},
    {"text": "CIFR", "start_char": 116, "end_char": 120}
  ],
  "sentiment_score": 0.3
}
```

### 2.3 Character Position Rules
- `start_char`: Index of first character (0-indexed)
- `end_char`: Index AFTER last character
- **Verification**: `text[start_char:end_char]` must EXACTLY match the mention text

---

## 3. Sentiment Scoring

### 3.1 Score Range
Score from **-1.0 to +1.0**:

| Score | Meaning | Examples |
|-------|---------|----------|
| -1.0 | Extremely negative | Bankruptcy, fraud, criminal charges |
| -0.7 to -0.9 | Very negative | Major earnings miss, mass layoffs |
| -0.4 to -0.6 | Moderately negative | Earnings miss, downgrade |
| -0.1 to -0.3 | Slightly negative | Minor miss, cautious outlook |
| 0.0 | Neutral | Factual reporting, passing mention |
| +0.1 to +0.3 | Slightly positive | Minor beat, stable outlook |
| +0.4 to +0.6 | Moderately positive | Earnings beat, upgrade |
| +0.7 to +0.9 | Very positive | Strong beat, raised guidance |
| +1.0 | Extremely positive | Record-breaking performance |

### 3.2 Sentiment Assignment Rules

**Entities WITH sentiment** (must have score between -1.0 and 1.0):
- ORG
- TICKER
- PERSON

**Entities WITHOUT sentiment** (must be `null`):
- MONEY
- PERCENT
- DATE

### 3.3 Context-Based Sentiment
**READ and UNDERSTAND** the article to determine sentiment. Do NOT use keyword matching.

**Example Analysis**:
```
"Cipher Mining Inc. (CIFR) Increases Despite Market Slip"
```
- The stock INCREASED (+1.17%)
- This happened DESPITE market decline (relative outperformance)
- Sentiment: **+0.3** (slightly positive - stock went up in a down market)

**NOT**: 0.0 just because there are no extreme keywords

### 3.4 Entity-Specific Sentiment
If article discusses multiple entities differently, assign different scores:

```
"Apple beat expectations while Microsoft missed badly."
```
- Apple: +0.5 (beat expectations)
- Microsoft: -0.5 (missed badly)

---

## 4. Coreference Mentions

### 4.1 Definition
Pronouns and generic references that refer to a specific entity:
- "it", "they" referring to a company
- "the company", "the firm", "the tech giant"
- "he", "she", "the CEO"

### 4.2 Placement
- `ner_mentions`: Proper entity names (used for NER training)
- `coref_mentions`: Pronouns and generic references (used for sentiment only)

**Example**:
```
"Apple reported strong earnings. The company expects continued growth."
```
```json
{
  "canonical_id": "AAPL",
  "canonical_name": "Apple Inc.",
  "type": "ORG",
  "ner_mentions": [{"text": "Apple", "start_char": 0, "end_char": 5}],
  "coref_mentions": [{"text": "The company", "start_char": 35, "end_char": 46}],
  "sentiment_score": 0.6
}
```

---

## 5. Output Format

### 5.1 JSON Schema
```json
{
  "id": "article_id",
  "text": "Full article text...",
  "entities": [
    {
      "canonical_id": "AAPL",
      "canonical_name": "Apple Inc.",
      "type": "ORG",
      "linked_ticker": "AAPL",
      "linked_company": null,
      "ner_mentions": [
        {"text": "Apple Inc.", "start_char": 0, "end_char": 10}
      ],
      "coref_mentions": [
        {"text": "the company", "start_char": 50, "end_char": 61}
      ],
      "sentiment_score": 0.5
    }
  ],
  "metadata": {
    "source": "...",
    "published_date": "..."
  }
}
```

### 5.2 Field Requirements

| Field | Required | Type | Notes |
|-------|----------|------|-------|
| canonical_id | Yes | string | Ticker if known, else entity name |
| canonical_name | Yes | string | Full/preferred entity name |
| type | Yes | string | ORG, TICKER, PERSON, MONEY, PERCENT, DATE |
| linked_ticker | No | string/null | For ORG: associated ticker |
| linked_company | No | string/null | For TICKER: associated company |
| ner_mentions | Yes | array | At least 1 mention |
| coref_mentions | No | array | Can be empty |
| sentiment_score | Yes | float/null | Required for ORG/TICKER/PERSON, null for others |

---

## 5.3 Two-Stage Labeling Schema (Extended)

For two-stage labeling, use this extended schema:

```json
{
  "canonical_id": "AAPL",
  "canonical_name": "Apple Inc.",
  "type": "ORG",
  "linked_ticker": "AAPL",
  "linked_company": null,
  "ner_mentions": [
    {"text": "Apple Inc.", "start_char": 0, "end_char": 10}
  ],
  "coref_mentions": [
    {"text": "the company", "start_char": 50, "end_char": 61},
    {"text": "it", "start_char": 100, "end_char": 102}
  ],
  "sentiment_expanded_mentions": [
    {"text": "tech giant", "start_char": 150, "end_char": 160}
  ],
  "sentiment_score": 0.5,
  "is_sentiment_only": false
}
```

### Field Definitions for Two-Stage

| Field | Stage | Description | Used For |
|-------|-------|-------------|----------|
| `ner_mentions` | 1 | Proper named entity mentions only | NER training |
| `coref_mentions` | 2 | Pronouns & generic references | Sentiment only |
| `sentiment_expanded_mentions` | 2 | Sector/index mentions | Sentiment only |
| `is_sentiment_only` | 2 | True if entity has no strict NER mentions | Filter flag |

### Entity Scope Rules

| Mention Type | What to Include | Training Use |
|--------------|-----------------|--------------|
| `ner_mentions` | Proper names, tickers, org names | NER model training |
| `coref_mentions` | "it", "they", "the company", "the firm" | Sentiment mask expansion |
| `sentiment_expanded_mentions` | "tech stocks", "S&P 500", "chipmakers" | Sentiment mask expansion (optional) |

### Sentiment-Only Entities

Some entities are relevant for sentiment but not for NER training:

```json
{
  "canonical_id": "TECH_SECTOR",
  "canonical_name": "Technology Sector",
  "type": "SECTOR",
  "ner_mentions": [],
  "coref_mentions": [],
  "sentiment_expanded_mentions": [
    {"text": "tech stocks", "start_char": 45, "end_char": 56}
  ],
  "sentiment_score": -0.3,
  "is_sentiment_only": true
}
```

These entities:
- Have `is_sentiment_only: true`
- Have empty `ner_mentions`
- Are used for sentiment analysis of market segments
- Should NOT be used for NER training

---

## 6. Common Mistakes to AVOID

### 6.1 Missing Company Names
❌ **WRONG**: Only extracting ticker `CIFR`
✅ **CORRECT**: Extract both `Cipher Mining Inc.` (ORG) AND `CIFR` (TICKER)

### 6.2 Neutral Sentiment Default
❌ **WRONG**: Defaulting to 0.0 when article has clear sentiment
✅ **CORRECT**: Read article and assign appropriate sentiment (+0.3 for "increases despite market slip")

### 6.3 Organization Acronyms as Tickers
❌ **WRONG**: Labeling `NYSE`, `SEC`, `ECB` as TICKER
✅ **CORRECT**: Label them as ORG (they are institutions, not stock symbols)

### 6.4 Incomplete Extraction
❌ **WRONG**: Only extracting the most prominent entity
✅ **CORRECT**: Extract ALL entities of all types mentioned in the article

### 6.5 Wrong Character Positions
❌ **WRONG**: Approximate or incorrect positions
✅ **CORRECT**: Exact positions where `text[start:end]` matches mention text

---

## 7. Quality Checklist

Before submitting labeled data, verify:

- [ ] ALL company names extracted as ORG
- [ ] ALL ticker symbols extracted as TICKER
- [ ] Ticker and company share same canonical_id when referring to same entity
- [ ] Character positions are exact: `text[start_char:end_char] == mention["text"]`
- [ ] Sentiment reflects article content, not just neutral default
- [ ] ORG/TICKER/PERSON have sentiment scores
- [ ] MONEY/PERCENT/DATE have `null` sentiment
- [ ] No organization acronyms (NYSE, SEC, etc.) labeled as TICKER

---

## 8. Example: Complete Labeling

**Input Article**:
```
Cipher Mining Inc. (CIFR) Increases Despite Market Slip: Here's What You Need to Know.
Cipher Mining Inc. (CIFR) reached $14.76 at the closing of the latest trading day,
reflecting a +1.17% change compared to its last close.
```

**Output**:
```json
{
  "id": "example_001",
  "text": "Cipher Mining Inc. (CIFR) Increases Despite Market Slip: Here's What You Need to Know. Cipher Mining Inc. (CIFR) reached $14.76 at the closing of the latest trading day, reflecting a +1.17% change compared to its last close.",
  "entities": [
    {
      "canonical_id": "CIFR",
      "canonical_name": "Cipher Mining Inc.",
      "type": "ORG",
      "linked_ticker": "CIFR",
      "linked_company": null,
      "ner_mentions": [
        {"text": "Cipher Mining Inc.", "start_char": 0, "end_char": 18},
        {"text": "Cipher Mining Inc.", "start_char": 87, "end_char": 105}
      ],
      "coref_mentions": [],
      "sentiment_score": 0.3
    },
    {
      "canonical_id": "CIFR",
      "canonical_name": "CIFR",
      "type": "TICKER",
      "linked_ticker": null,
      "linked_company": "Cipher Mining Inc.",
      "ner_mentions": [
        {"text": "CIFR", "start_char": 20, "end_char": 24},
        {"text": "CIFR", "start_char": 107, "end_char": 111}
      ],
      "coref_mentions": [],
      "sentiment_score": 0.3
    },
    {
      "canonical_id": "$14.76",
      "canonical_name": "$14.76",
      "type": "MONEY",
      "linked_ticker": null,
      "linked_company": null,
      "ner_mentions": [
        {"text": "$14.76", "start_char": 121, "end_char": 127}
      ],
      "coref_mentions": [],
      "sentiment_score": null
    },
    {
      "canonical_id": "+1.17%",
      "canonical_name": "+1.17%",
      "type": "PERCENT",
      "linked_ticker": null,
      "linked_company": null,
      "ner_mentions": [
        {"text": "+1.17%", "start_char": 185, "end_char": 191}
      ],
      "coref_mentions": [],
      "sentiment_score": null
    },
    {
      "canonical_id": "latest_trading_day",
      "canonical_name": "latest trading day",
      "type": "DATE",
      "linked_ticker": null,
      "linked_company": null,
      "ner_mentions": [
        {"text": "latest trading day", "start_char": 147, "end_char": 165}
      ],
      "coref_mentions": [],
      "sentiment_score": null
    }
  ],
  "metadata": {}
}
```

**Key Points**:
- Both ORG and TICKER extracted for the company
- Same canonical_id links them
- Sentiment +0.3 reflects positive news (stock increased)
- All entity types captured
- Exact character positions

---

## 9. spaCy Validation Integration

### 9.1 Purpose
Use spaCy transformer model as a cross-check for LLM entity extraction to catch missed entities.

### 9.2 Validation Process
1. Run LLM labeling on article
2. Run spaCy NER on the same article text
3. Compare entities found by each
4. Flag misalignments for review

### 9.3 Handling Misalignments
When spaCy finds entities that LLM missed:
1. Check if spaCy entity is valid per this spec
2. If valid and LLM missed it → Add to output
3. If spaCy is wrong (e.g., "F" in "ETF" as ORG) → Ignore spaCy

When LLM finds entities that spaCy missed:
- Usually correct (LLM has better context understanding)
- Keep LLM result unless clearly wrong

### 9.4 spaCy Entity Type Mapping
| spaCy Label | Our Label | Notes |
|-------------|-----------|-------|
| ORG | ORG | Direct mapping |
| PERSON | PERSON | Direct mapping |
| MONEY | MONEY | Direct mapping |
| PERCENT | PERCENT | Direct mapping |
| DATE | DATE | Direct mapping |
| GPE | ORG | Geopolitical entities (EU, US) → treat as ORG if relevant |
| NORP | ORG | National/religious/political groups → ORG if organization |

### 9.5 Validation Script Usage
```bash
python scripts/labeling/validate_with_spacy.py --input labeled_articles.jsonl --output validated_articles.jsonl
```

---

## 10. Known Edge Cases

### 10.1 Single-Letter Tickers
- `F` (Ford), `X` (US Steel), `V` (Visa), `T` (AT&T)
- ALWAYS verify context before labeling
- Check for explicit company mention or stock discussion

### 10.2 Ambiguous Acronyms
- `AI` - could be ticker (C3.ai) or concept (artificial intelligence)
- `META` - ticker (Meta Platforms) or generic term
- Use article context to determine

### 10.3 Partial Company Names
- "Monster" in "Monster Beverage" article → still extract as ORG
- "Apple" vs "apple" (fruit) → context determines if it's the company

### 10.4 Governmental Bodies
- EU, UN, NATO → Always ORG when discussing policy/regulation impact
- These affect markets and should have sentiment scores
