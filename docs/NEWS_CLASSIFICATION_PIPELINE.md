# News Classification Pipeline (Phase 2)

**Last updated:** 2026-05-24
**Audience:** Future Russell, Claude, Kimi, or any agent helping with this project.
**Goal:** Take the raw 1.69M-article EODHD news corpus from Phase 1 and tag every article with a source **tier (T1/T2/T3)** plus a fine-grained **content_type**, producing the labeled corpus the sentiment model uses for loss weighting.

This pipeline runs *after* Phase 1 (`scripts/collection/collect_eodhd_sp500_bulk.py`) and *before* Phase 3 (build training subsets).

---

## TL;DR — How to re-run the full Phase 2

You have a fresh `data/raw/<bulk_dir>/news/*.jsonl.gz` directory from Phase 1 and want every article retiered. Six steps, ~50 min total wall time, ~$0 (uses Kimi For Coding subscription).

```bash
BULK=data/raw/eodhd_bulk_20260518   # ← change to your Phase 1 output dir

# 1. Stratified 5K sample (density × year quintiles, ~30 sec)
python3 scripts/analysis/sample_for_source_classification.py \
    --input-dir $BULK/news \
    --output outputs/source_classification_sample.jsonl \
    --n-samples 5000 --seed 42

# 2. Classify the 5K sample with Kimi K2.6 (~22 min, 4 workers)
python3 scripts/labeling/classify_sources_haiku.py \
    --input outputs/source_classification_sample.jsonl \
    --output outputs/source_classification_kimi.jsonl \
    --provider kimi --workers 4 --batch-size 50

# 2b. Reclassify the "other" bucket once the extended enum is in place (~12 min)
#     Extract → reclassify → merge. See §3.2 for details.

# 3. Aggregate Kimi labels into raw rule candidates (~1 min)
python3 scripts/analysis/aggregate_source_classification.py \
    --input outputs/source_classification_kimi_v2.jsonl \
    --output outputs/candidate_source_rules.json --min-support 5

# 4. Validate the production rule set against Kimi labels (~10 sec)
python3 scripts/analysis/validate_source_rules.py \
    --validation-file outputs/source_classification_kimi_v2.jsonl \
    --rules-file outputs/source_rules_v4.json \
    --output outputs/rule_validation_report_v4.json

# 5. Dry-run on 1% of the corpus to detect drift (~1 min)
python3 scripts/preprocessing/dry_run_source_tiers.py \
    --input-dir $BULK/news \
    --rules-file outputs/source_rules_v4.json \
    --sample-rate 0.01 --seed 42

# 6. Apply rules to the full corpus (~25 min)
python3 scripts/preprocessing/apply_source_tiers.py \
    --input-dir $BULK/news \
    --output-dir $BULK/news_retiered_v4 \
    --rules-file outputs/source_rules_v4.json
```

Result: `$BULK/news_retiered_v4/*.jsonl.gz` — each record now carries `rule_name`, `final_source_tier`, `content_type`, `detected_source`, `syndication_source`. Downstream training reads `final_source_tier` for loss weighting.

---

## What "tier" means and why it exists

The sentiment model weights training loss by source quality. Wire-grade content (Reuters, Bloomberg) is treated as ground truth; algo-generated stock bulletins are downweighted because their sentiment signal is noisy.

| Tier | Examples | Loss weight | Share of 1.69M corpus |
|---|---|---:|---:|
| **T1** — wire-grade | Bloomberg, Reuters, AP, WSJ, FT, Barron's, MarketWatch | 1.0 | **9.62%** |
| **T2** — analytical | Investing.com, IBD, Yahoo Market Wrap, Yahoo Analyst Call, Earnings Transcripts, Benzinga, MT Newswires | 0.5 | **9.04%** |
| **T3** — algo / SEO / press release | Zacks, Motley Fool, GuruFocus, Insider Monkey, Seeking Alpha, Simply Wall St, GlobeNewswire/PR Newswire/Business Wire, Yahoo SEO listicles | 0.2 | **81.34%** |

The 81% T3 share reflects EODHD's actual composition — Yahoo Finance pages are dominated by SEO/promotional content. T3 isn't useless (it provides recall and entity coverage); it's just trusted less per-article.

---

## The 6 steps in detail

### Step 1 — Stratified sample (`sample_for_source_classification.py`)

Picks 5,000 articles to label with Kimi.

- **Stratification:** by (Phase-1 source tier × density quintile × year) — ensures coverage of rare sources and rare years (2020 is sparse: ~91 NVDA articles in 2020 vs 41K in 2025).
- **Density-weighted within each stratum:** `p ∝ 1 / sqrt(N_ticker)` to upweight rare-news tickers.
- **Output:** `outputs/source_classification_sample.jsonl` (~6 MB).

The sample is deterministic via `--seed 42`. If you re-pull articles, you can re-sample with the same seed and get a similar (not identical) distribution.

### Step 2 — LLM classification (`classify_sources_haiku.py`)

Sends each article (title + first 500 chars) to Kimi K2.6 (`kimi-for-coding` model) via the Anthropic SDK pointed at `https://api.kimi.com/coding`.

- **Provider:** Kimi For Coding subscription. The Anthropic SDK sends `x-stainless-*` headers that Moonshot accepts; direct OpenAI-style calls return 403. Set `--provider anthropic` to fall back to Claude Haiku 4.5 if needed.
- **Auth:** reads `api.kimi_api_key` from `config/secrets.yaml` (or `KIMI_API_KEY` env var).
- **Workers:** 4. Moonshot's per-account limit is 1200 req/min; we run well under at ~120 req/min.
- **Backoff:** exponential 2s/4s/8s with ±25% jitter on 4xx/5xx. Permanent 400 "high risk" failures (geopolitical content) are accepted as a small unclassifiable residual (typically 0.1–0.2% of articles).
- **Output:** `outputs/source_classification_kimi.jsonl` — adds `haiku_classification: {content_type, confidence, other_description}` and `classified_at`, `classifier_model` to each record.

**Enum** (25 categories): `bloomberg`, `reuters`, `ap`, `wsj`, `ft`, `barrons`, `marketwatch`, `mt_newswires`, `motleyfool`, `seeking_alpha`, `ibd`, `investing_com`, `benzinga`, `barchart`, `earnings_transcript`, `gurufocus`, `insider_monkey`, `simplywallst`, `zacks_press_release`, `yahoo_market_wrap`, `yahoo_stock_bulletin`, `yahoo_analyst_call`, `yahoo_seo_promotional`, `press_release_other`, `other`.

#### Step 2b — Reclassify the "other" bucket

After the first pass typically ~19% of articles land in "other". Inspect the `other_description` field to see what's being missed; if there are new named sources with ≥20 articles each, add them to the enum and reclassify only the "other" subset.

```bash
# Extract "other" articles
python3 -c "
import json
src='outputs/source_classification_kimi.jsonl'
dst='outputs/source_classification_other_for_reclassify.jsonl'
n=0
with open(src) as f, open(dst,'w') as out:
    for line in f:
        r=json.loads(line)
        if (r.get('haiku_classification') or {}).get('content_type')=='other':
            r.pop('haiku_classification',None); r.pop('classified_at',None); r.pop('classifier_model',None)
            out.write(json.dumps(r,ensure_ascii=False)+'\n'); n+=1
print(f'wrote {n} other articles to {dst}')
"

# Reclassify them (with the extended enum in classify_sources_haiku.py)
python3 scripts/labeling/classify_sources_haiku.py \
    --input outputs/source_classification_other_for_reclassify.jsonl \
    --output outputs/source_classification_other_reclassified.jsonl \
    --provider kimi --workers 4 --batch-size 50

# Merge back, producing the canonical kimi_v2 sample
python3 -c "
import json
recl={json.loads(l)['id']: json.loads(l) for l in open('outputs/source_classification_other_reclassified.jsonl')}
with open('outputs/source_classification_kimi.jsonl') as fin, open('outputs/source_classification_kimi_v2.jsonl','w') as fout:
    for line in fin:
        r=json.loads(line)
        fout.write(json.dumps(recl.get(r['id'],r),ensure_ascii=False)+'\n')
"
```

After our May-2026 run, "other" shrank from 19.29% → 6.62% via 11 new buckets (barrons, mt_newswires, seeking_alpha, ibd, investing_com, benzinga, barchart, earnings_transcript, gurufocus, insider_monkey, simplywallst).

### Step 3 — Aggregate into candidate rule signals (`aggregate_source_classification.py`)

For each `content_type` with ≥`--min-support 5` articles, extracts:
- Top exact prefixes of length 50, 100
- Longest common prefix across all articles in that category
- Distinctive 2-grams and 3-grams (frequent in this category, rare elsewhere)
- 5 sample titles

**Output:** `outputs/candidate_source_rules.json`. This is a *signal report*, not a ready-to-use rule file. Use it to inform manual regex design in Step 4.

### Step 4 — Hand-author and validate the rule set (`validate_source_rules.py`)

The actual production rules live in `outputs/source_rules_v4.json` (35 rules). They're hand-authored from the Step 3 signals plus domain knowledge (publication URL patterns, bylines, paywall stubs).

**Rule schema** (each rule is a JSON object with these fields):
```json
{
  "name": "reuters_url",
  "pattern": "(?:^|\\n)\\s*(?:https?://)?(?:www\\.)?reuters\\.com\\b",
  "target_content_type": "reuters",
  "target_tier": 1,
  "detected_source": "Reuters",
  "content_type": "reuters",
  "syndication_source": "Reuters"
}
```

The duplicated `target_content_type`/`content_type` and `target_tier`/`tier` fields let both the validator and the applier load the same JSON.

**Matcher** (in both `validate_source_rules.py` and `apply_source_tiers.py`):
```
if pattern.startswith("^"):
    text = content                     # body-prefix rules
else:
    body = (title + "\n" + content)[:1000]
    text = f"{url}\n{url_domain}\n{body}"   # URL fields prepended
return bool(re.compile(pattern, re.IGNORECASE).search(text))
```

The 1000-char body cap matters — late-article footers (e.g., "Most Read from Bloomberg" appearing 5000 chars deep in a Reuters article) caused FPs in Phase 1; the cap prevents the regression.

**Rule ordering matters — first match wins.** Current order:
1. T1 wire prefixes/URLs (bloomberg, reuters, ap)
2. T1 brand patterns (wsj, ft, barrons, marketwatch)
3. T2 mt_newswires (demoted from T1 on 2026-05-31 — EODHD only carries paywall stubs; see note below)
4. T2/T3 specific-source rules (ibd, yahoo_market_wrap, investing_com, yahoo_analyst_call, earnings_transcript, benzinga, barchart, gurufocus, insider_monkey, simplywallst, motleyfool)
5. T3 zacks rules
6. T3 press-release distributor rules (globenewswire, prnewswire, businesswire, accesswire, newsfile)
7. T3 seeking_alpha (placed **late** because SA hosts diverse content; specific source-attribution rules should claim their articles first)
8. T3 yahoo_seo_listicle (generic title-based catch-all, placed last)

**When you change rules**, always re-validate against the Kimi sample to confirm tier accuracy doesn't regress:

```bash
python3 scripts/analysis/validate_source_rules.py \
    --validation-file outputs/source_classification_kimi_v2.jsonl \
    --rules-file outputs/source_rules_v4.json \
    --output outputs/rule_validation_report_v4.json
```

Acceptable targets:
- **Tier accuracy ≥ 87%** on the Kimi sample
- **Zero false T1 promotions** from T3 ground truth
- No per-rule precision below 50% (low-precision rules are stealing matches)

### Step 5 — Dry-run on 1% (`dry_run_source_tiers.py`)

Before applying to 1.69M, sample 1% of the corpus uniformly random and check the rule-match distribution + tier distribution match the Kimi sample. Detects **stratification drift** — if the full corpus has materially different composition than the 5K sample, you'll see large drifts in the per-category table.

The dry-run also prints the **Phase-1 fallback tier distribution** for `_no_rule` articles. ~92% of unmatched articles should already be Phase-1 T3 (correctly low-quality); higher than that and Phase-1's regex tiers may be miscalibrated.

If T1/T2/T3 distribution differs from the Kimi sample by more than 2 percentage points per tier, investigate before proceeding. Acceptable causes: sample stratification oversampled "easy" content (which is expected — we density-weighted). Unacceptable causes: a rule fires far more often at scale than in the sample (suggests adversarial body content or pattern over-match).

### Step 6 — Apply at scale (`apply_source_tiers.py`)

Streams every `<TICKER>.jsonl.gz`, applies rules in order, writes atomic gzipped output to `--output-dir`. Each record gets:
- `rule_name` — the rule that matched, or `null` if no rule fired
- `final_source_tier` — T1/T2/T3 (from rule, or Phase-1 fallback `record["source_tier"]` if no rule matched)
- `content_type` — fine-grained category (or `"other"` for no_rule)
- `detected_source` — human-readable source name
- `syndication_source` — for wire-syndicated content

**Wall time:** ~25 min for 1.69M articles, single-process. Output dir is ~8% larger than input due to added fields.

---

## Rule design philosophy (why v4 is the way it is)

After three iterations (v2 → v3 → v4) with Kimi as a critic, the principles that stuck:

1. **Tier accuracy beats category accuracy.** Misclassifying Motley Fool as `yahoo_seo_promotional` is fine — both are T3 — but misclassifying Reuters as T3 is not. Optimize the rule set for tier-level confusion, not category-level F1.

2. **Phase-1 fallback is your safety net.** ~56% of articles end up in `_no_rule` — but Phase-1's URL-based tier (mostly correct for Yahoo no-attribution content → T3) is preserved. Don't try to write a Phase-2 rule for every article; rely on Phase-1 for the long tail.

3. **URL fields are higher-quality signal than body content.** Yahoo Finance laundering removes in-body attribution for ~111 MW/FT/Barrons/Benzinga articles in our sample; those are unrecoverable. But direct wire URLs (`reuters.com`, `bloomberg.com`, `apnews.com`) survive intact. URL-based rules with line anchors `(?:^|\n)` are precise enough for T1 promotion.

4. **First-match-wins ordering is a feature, not a bug.** Place specific source attributions (insider_monkey, earnings_transcript, press_release_other variants) **before** broad domain catches (seeking_alpha_story). Seeking Alpha hosts everyone's content; let the underlying source claim its articles first.

5. **Drop rules whose precision falls below 50% unless they're tier-safe.** Example: `bloomberg_footer` had P=45% in v3 because random articles cite "Bloomberg News reports". We narrowed to `Most Read from Bloomberg` (the actual SA-rehosted Bloomberg footer signal), bringing precision to 100%.

6. **The 1000-char body cap is a defensive design, not a limitation.** Footer scrapers (Zacks, Bloomberg, GuruFocus) place attribution at the end of articles; matching that footer in the *middle* of a Reuters article (when their footers happen to mention Zacks in a sponsored sidebar) caused FPs in Phase 1. The cap costs us some recall on late-attribution patterns but keeps precision clean.

7. **Tier weights at training time can flatten residual rule errors.** Even a 13% tier-error rate gets blunted because T3's loss weight is 0.2 (vs T1's 1.0). A T1 article wrongly tagged T3 contributes 20% of the loss it should — not zero. Don't chase the last 1% of tier accuracy at the cost of brittle, narrowly-tuned rules.

---

## Production metrics (v4 ruleset)

### On the Kimi-labeled 4,962-article evaluation set

- **Tier-level accuracy:** 87.36%
- **Overall precision:** 86.78%
- **Overall recall:** 33.66% (low because Phase-1 fallback handles most articles)
- **False T1 promotions from genuine T3:** 36 articles / 3,864 (0.93%) — all are `reuters.com` URLs Kimi mis-labeled as "other", so tier-wise correct

### On the full 1.69M corpus (May 2026 run)

| Tier | Count | Share |
|---|---:|---:|
| T1 | 162,659 | 9.62% |
| T2 | 152,890 | 9.04% |
| T3 | 1,375,466 | 81.34% |

| Top T1 source | Articles | Rule |
|---|---:|---|
| Reuters | 97,310 | `reuters_attribution` + `reuters_url` + `reuters_prefix` |
| Bloomberg | 39,917 | `bloomberg_prefix` + `bloomberg_footer` + `bloomberg_url` |
| ~~MT Newswires~~ | ~~14,530~~ | `mt_newswires` — **demoted T1→T2 on 2026-05-31** (paywall stubs); counts reflect the pre-demotion corpus. The next re-tier will move these ~14,530 articles to T2. |
| AP | 6,102 | `ap_dateline` + `ap_url` |
| WSJ | 1,549 | `wsj_url_or_brand` |
| Barron's | 209 | `barrons_brand` |
| FT | 160 | `ft_url` |

> **Note (2026-05-31):** The tier-share and source-count figures above reflect the `news_retiered_v4` corpus *before* the MT Newswires demotion. Because the existing corpus has not been re-tiered (the daily-aggregation step filters `mt_newswires` analysis-side, making a full re-tier redundant until the next bulk pull), these counts are accurate for the current data files. A future re-tier would shift ~14,530 articles from T1 (9.62% → ~8.76%) to T2.
| MarketWatch | 44 | `marketwatch_url_or_brand` |

The low MW/FT/Barrons capture rate (∼5% of true count in the Kimi sample) is unavoidable: Yahoo Finance strips publisher attribution when rehosting these paywalled publishers. Their articles fall through to `_no_rule` and inherit Phase-1's tier (typically T3 for Yahoo URLs without other signals).

---

## When to re-run vs iterate

**Re-run Steps 1–6 from scratch:**
- New Phase-1 collection (e.g., quarterly refresh, new ticker universe)
- Major source-mix shift (e.g., EODHD changes its content providers)

**Iterate Step 4 only (add/tune rules, re-validate, re-apply):**
- Spot-check finds systematic mis-classification of a specific source
- New publisher emerges in the 1.69M corpus that isn't in v4
- Downstream training reveals tier-weighting is too aggressive/conservative

**Skip Phase 2 entirely:**
- Hot-fix data quality bug — re-apply v4 to a different `--input-dir`. v4 is stable as long as the Phase-1 record schema (`title`, `content`, `url`, `url_domain`, `source_tier`) is unchanged.

---

## Files this pipeline produces and consumes

### Inputs
- `data/raw/<bulk_dir>/news/*.jsonl.gz` — Phase 1 output
- `config/secrets.yaml` — `api.kimi_api_key`
- `outputs/source_rules_v4.json` — production rule set (in git)

### Outputs
- `outputs/source_classification_sample.jsonl` — 5K stratified sample
- `outputs/source_classification_kimi.jsonl` — Kimi first-pass labels
- `outputs/source_classification_other_reclassified.jsonl` — Kimi second-pass labels for the "other" bucket
- `outputs/source_classification_kimi_v2.jsonl` — canonical merged labels (use this for validation)
- `outputs/candidate_source_rules.json` — Step 3 raw signals
- `outputs/rule_validation_report_v4.json` — Step 4 P/R/F1 + tier confusion
- `data/raw/<bulk_dir>/news_retiered_v4/*.jsonl.gz` — Step 6 final corpus

### Scripts (all in repo)
- `scripts/analysis/sample_for_source_classification.py`
- `scripts/labeling/classify_sources_haiku.py`
- `scripts/analysis/aggregate_source_classification.py`
- `scripts/analysis/validate_source_rules.py`
- `scripts/preprocessing/dry_run_source_tiers.py`
- `scripts/preprocessing/apply_source_tiers.py`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Step 2: HTTP 403 from Kimi | Direct OpenAI-style calls hit the "coding agents only" gate | The script uses Anthropic SDK protocol; if you've modified it, restore `base_url="https://api.kimi.com/coding"` and call via `anthropic.Anthropic` |
| Step 2: many "high risk" 400s | Geopolitical content triggering content filter | Expected at 0.1–0.2%; failures retried 3× then accepted. Fall back to `--provider anthropic` for the failed articles if needed |
| Step 2: very slow (<1 batch/min) | Workers too high causing rate-limit backoff loop | Drop to `--workers 2`, raise `--batch-size` to 100 |
| Step 4: a rule scores 0% precision and 0% recall | Pattern doesn't match anywhere in the matcher's `url + url_domain + title + content[:1000]` text | Use the diagnostic snippet in Step 4 to check which fields the target articles actually contain |
| Step 4: tier accuracy drops after rule edit | New rule is stealing matches from a higher-precision rule | Check rule ordering; specific-source rules go before broad domain catches |
| Step 5 dry-run: T1/T2/T3 distribution drifts >2% from Kimi sample | Phase-1 fallback tier is wrong for many no_rule articles, OR a rule fires more often at scale due to body-text adversarial matches | Inspect the per-rule sample counts; if one rule jumped >2× its sample rate, tighten its pattern |
| Step 6: only some files written before exit | Disk full, or interrupted | Re-run; the applier writes atomic gzip (temp file + rename), so partial files are safe to retry. Skip already-written files manually if needed |

---

## Acknowledgments

The v4 ruleset is the product of three iterative rounds with Kimi K2.6 as second-opinion reviewer (May 2026). Key contributions from that review: reordering `seeking_alpha_story` after body-attribution rules, tightening `insider_monkey` and `barchart`, adding the `yahoo_market_wrap` title catcher, and identifying the Yahoo-laundering ceiling at ~88–90% tier accuracy.
