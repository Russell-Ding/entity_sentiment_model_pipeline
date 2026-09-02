# signal_strategy/

Exploration of whether v2.1 model-derived entity/ticker sentiment can drive a stock-trading signal.

## Layout
- `RESULTS.md` — **first-pass findings (start here).**
- `src/` — pipeline: `build_panel.py`, `ic_analysis.py`, `fama_macbeth.py`,
  `event_study.py`, `backtest_ls.py`, shared `data_utils.py`, `config.yaml`,
  and `run_all.py` (runner).
- `outputs/` — `panel.parquet`, metrics `*.json/*.csv`, and plots
  (`ic_by_horizon.png`, `event_study_car.png`, `ls_equity_curve.png`).
- `research/` — prior-work review and references.
  - `literature_review.md` — survey of news-sentiment→returns research + signal-construction methods, with a recommended first model to test.
- `notes/` — working notes, ideas, scratch.

## Reproduce
```bash
python3 signal_strategy/src/run_all.py          # full pipeline (~7 min; Step 1 dominates)
python3 signal_strategy/src/run_all.py --skip-panel   # reuse panel.parquet (~1 min)
```

## Bottom line
The signal **explains returns; it does not predict them.** The first-pass
"predictive but untradable" read was look-ahead: ~25% of articles (incl.
after-hours earnings) are assigned a trade_date whose close precedes publication.
With an honest 15:45 ET knowability cutoff (`moc_feasibility.py`,
`panel_feasible.parquet`) the IC is ≈ 0 at every horizon and every daily-bar
execution is negative. Two things survive: the model is a good *measurement*
instrument (attribution/monitoring), and sentiment **magnitude** predicts
next-day **volatility** beyond trailing vol (rank-IC +0.024, NW t 5.3).
Externally reviewed by Codex and Kimi (`notes/reviews/`); Kimi independently
re-ran the key numbers and confirmed both the look-ahead diagnosis and the
surviving vol result. See `RESULTS.md`.

## Data ingredients (generated locally from EODHD data; not redistributed in the public repo)
- **Sentiment:** `outputs/inference/*_enriched.jsonl` — per-article entity + `ticker_sentiments`, with `trade_date` (501 tickers, ~2021–2025).
- **Prices:** `data/raw/eodhd_bulk_20260518/prices/*.csv` — daily OHLCV + `adjusted_close`.
- **Model:** `trained_model/v2.1_20260620/` — TICKER sentiment strongest (r≈0.81), best suited for ticker→returns.

## Recommended first step (from the review)
Build a per-ticker daily sentiment panel → check Information Coefficient + Fama-MacBeth → event-study to pick the holding window → quintile long-short with next-open execution and costs/turnover reported. Detect and size the effect before optimizing a backtest.
