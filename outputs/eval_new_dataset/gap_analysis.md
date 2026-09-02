# Gap Analysis — current model vs new DeepSeek+Kimi labels

Covered entities: **164,723** (from 1 prediction file(s))

## Headline correlations

| scope | N | Pearson r [95% CI] | Spearman ρ [95% CI] | MAE |
|---|---:|---|---|---:|
| **overall** | 164,723 | +0.485 [+0.480, +0.489] | +0.451 [+0.447, +0.456] | 0.209 |
| ORG | 126,865 | +0.486 [+0.481, +0.491] | +0.446 [+0.440, +0.451] | 0.208 |
| TICKER | 2,109 | +0.622 [+0.591, +0.650] | +0.623 [+0.593, +0.651] | 0.254 |
| PERSON | 35,749 | +0.493 [+0.484, +0.501] | +0.470 [+0.461, +0.479] | 0.209 |

## Trivial baseline
- predict-the-mean (+0.013) MAE = **0.231**; model MAE = **0.209** → model beats the mean predictor.
- a constant predictor has Pearson r undefined (0); any positive r beats it. MAE comparison shows whether the model beats predicting the mean.

## Systematic gaps
- **Over-neutrality**: of 53,983 entities with gold |s|≥0.4, the model is ~flat (|pred|<0.1) on **16,625** (**30.8%**) — the lawsuit/probe blind spot.
- **Sign flips**: among 61,310 both-signed entities, **13,160** flip sign (**21.5%**).

### Error by gold magnitude
| band | N | MAE | mean gold | mean pred |
|---|---:|---:|---:|---:|
| incidental | 71,028 | 0.090 | +0.001 | +0.014 |
| mild | 39,712 | 0.222 | +0.024 | +0.030 |
| moderate | 42,613 | 0.333 | +0.006 | +0.020 |
| strong | 11,370 | 0.440 | +0.069 | +0.078 |

## Span contamination check (DeepSeek spans, not human-gold)
- matched-span IoU: mean 0.968, median 1.000; 6% of matches have IoU<0.7 (partial-overlap → possibly different sentiment target).
- 75,970/526,105 (14%) predicted sentiment-spans matched no gold entity.

## PERSON: Kimi-rescored vs not
| subset | N | Pearson r | Spearman ρ |
|---|---:|---|---|
| PERSON_kimi_rescored | 34,415 | +0.497 [+0.487, +0.506] | +0.476 [+0.467, +0.485] |
| PERSON_not_rescored | 1,334 | +0.289 [+0.224, +0.354] | +0.234 [+0.177, +0.288] |

## Bucket confusion (exact acc 56.2%)
| gold ↓ / pred → | very_neg | neg | neutral | pos | very_pos |
|---|---|---|---|---|---|
| very_neg | 0 | 553 | 1017 | 63 | 0 |
| neg | 0 | 9399 | 23330 | 1972 | 0 |
| neutral | 0 | 2771 | 70065 | 5555 | 1 |
| pos | 0 | 1990 | 28521 | 13105 | 16 |
| very_pos | 0 | 231 | 2165 | 3944 | 25 |

## Per-company (top by frequency)
| entity | N | Pearson r [CI] |
|---|---:|---|
| TSLA | 7,458 | +0.520 [+0.502, +0.538] |
| GOOGL | 7,136 | +0.569 [+0.549, +0.587] |
| AAPL | 7,004 | +0.482 [+0.461, +0.504] |
| AMZN | 6,204 | +0.568 [+0.548, +0.587] |
| MSFT | 5,983 | +0.490 [+0.469, +0.513] |
| NVDA | 4,167 | +0.533 [+0.509, +0.557] |
| META | 4,028 | +0.510 [+0.482, +0.536] |
| JPM | 2,620 | +0.567 [+0.536, +0.601] |
| OPENAI | 1,215 | +0.465 [+0.413, +0.516] |
| TWTR | 1,204 | +0.350 [+0.295, +0.406] |
| GS | 1,126 | +0.382 [+0.310, +0.445] |
| GM | 1,063 | +0.405 [+0.346, +0.464] |
| ELON_MUSK | 969 | +0.561 [+0.519, +0.602] |
| FED | 965 | +0.227 [+0.173, +0.283] |
| INTC | 949 | +0.272 [+0.195, +0.347] |

### Over-neutral examples
- **MDT** (ORG) gold=0.4 pred=0.061 — _Apple Partner Foxconn to Start Making Ventilators in U.S.. (Bloomberg) -- Foxconn, the company responsible for assembling most of the world’s Apple Inc. iPhones, will aid the fight against the coronavirus pandemic by developing and making v…_
- **Omar Ishrak** (PERSON) gold=0.4 pred=0.023 — _Apple Partner Foxconn to Start Making Ventilators in U.S.. (Bloomberg) -- Foxconn, the company responsible for assembling most of the world’s Apple Inc. iPhones, will aid the fight against the coronavirus pandemic by developing and making v…_
- **FANUC** (ORG) gold=0.5 pred=-0.056 — _The Robots-Are-Taking-Our-Jobs Threat Gets Real. (Bloomberg Opinion) -- If there was ever a good time for the robots-taking-over-jobs argument, this may be it. Not just because factory owners don’t want to pay for rising labor costs, but be…_
- **HARMONIC** (ORG) gold=0.4 pred=-0.03 — _The Robots-Are-Taking-Our-Jobs Threat Gets Real. (Bloomberg Opinion) -- If there was ever a good time for the robots-taking-over-jobs argument, this may be it. Not just because factory owners don’t want to pay for rising labor costs, but be…_
- **AAPL** (ORG) gold=-0.4 pred=-0.035 — _Little-Known Data Show Signs of a Tech Bounce. (Bloomberg Opinion) -- Buried in a set of little-known data are early signs that the hardware side of the technology sector may be rebounding from the pandemic-driven plunge.  Investors general…_
- **FOXCONN** (ORG) gold=-0.4 pred=-0.03 — _Little-Known Data Show Signs of a Tech Bounce. (Bloomberg Opinion) -- Buried in a set of little-known data are early signs that the hardware side of the technology sector may be rebounding from the pandemic-driven plunge.  Investors general…_
- **TSM** (ORG) gold=0.6 pred=-0.036 — _Little-Known Data Show Signs of a Tech Bounce. (Bloomberg Opinion) -- Buried in a set of little-known data are early signs that the hardware side of the technology sector may be rebounding from the pandemic-driven plunge.  Investors general…_
- **QUANTA** (ORG) gold=0.4 pred=-0.063 — _Little-Known Data Show Signs of a Tech Bounce. (Bloomberg Opinion) -- Buried in a set of little-known data are early signs that the hardware side of the technology sector may be rebounding from the pandemic-driven plunge.  Investors general…_

### Sign-flip examples
- **KWEICHOW_MOUTAI** (ORG) gold=-0.3 pred=0.305 — _Crazy Trading on China's Nasdaq Has Its Own Logic. (Bloomberg Opinion) -- Old habits die hard. China’s Nasdaq-like ChiNext board has staged a comeback this year, rallying 17% to trade at an average of — yes, really — 70 times earnings. Is t…_
- **AAPL** (ORG) gold=0.2 pred=-0.215 — _Apple to Suspend New Orders to Wistron After India Workers Riot. (Bloomberg) -- Apple Inc. said it’s placing a supplier in India on probation after lapses in labor practices led to rioting, and will hold off giving new orders to Wistron Cor…_
- **WDC** (ORG) gold=0.3 pred=-0.27 — _Samsung Will Offer Clues on How Covid-19 Is Roiling Global Tech. (Bloomberg) --  When Samsung Electronics Co. brass addressed analysts during its last earnings call, much of the talk revolved around finally turning the corner after years in…_
- **MU** (ORG) gold=0.4 pred=-0.291 — _Samsung Will Offer Clues on How Covid-19 Is Roiling Global Tech. (Bloomberg) --  When Samsung Electronics Co. brass addressed analysts during its last earnings call, much of the talk revolved around finally turning the corner after years in…_
- **AAPL** (TICKER) gold=0.3 pred=-0.141 — _Olympics delay deals setback to Samsung's plans to win over Japan market. By Hyunjoo Jin  SEOUL (Reuters) - For Samsung Electronics Co Ltd <005930.KS>, the 2020 Tokyo Olympics were going to be its springboard to attain a long-held goal - ma…_
- **AAPL** (ORG) gold=-0.3 pred=0.387 — _Apple's $549 AirPods Max Headphones Offer Big Sound and Bugs: Review. (Bloomberg) -- The AirPods Max headphones sound terrific. For almost the price of an iPhone 11, anything less would be a major letdown.  Apple Inc.’s entry into the rapid…_
- **TRANSSION** (ORG) gold=0.4 pred=-0.177 — _China in Africa Is More Than a Land Grab. (Bloomberg Opinion) -- China has plenty to gain from lending a hand to its friends battling the coronavirus in Africa. Contrary to some perceptions, that won't mean opportunistic grabs in oil, coppe…_
- **TSLA** (ORG) gold=-0.5 pred=0.236 — _An Apple-Tesla Showdown Will Happen on the Factory Floor. (Bloomberg Opinion) -- It was kind of inevitable that Tesla Inc.’s biggest challenger wouldn’t be a car company. Apple Inc. makes for the perfect nemesis and could teach its Californ…_
