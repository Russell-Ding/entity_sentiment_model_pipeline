You are an external reviewer for a completed quant research study in this repo. This is a STRICTLY READ-ONLY review — do not modify, create, or delete any files.

Read, in order:
1. signal_strategy/RESULTS.md (the findings)
2. signal_strategy/src/ — config.yaml, data_utils.py, build_panel.py, ic_analysis.py, fama_macbeth.py, event_study.py, backtest_ls.py, moc_feasibility.py
3. Outputs if useful: signal_strategy/outputs/*.json

Context: a fine-tuned entity-sentiment model (v2.1) scores financial news per ticker (mention-weighted sent_mw). The study tests whether this predicts cross-sectional stock returns for ~500 large caps, 2020-2025, daily bars. Claimed findings:
(1) Real predictive content: 1d IC IR ~0.10 (NW t=3.65); Fama-MacBeth ~10bps/SD at 1-5d (t=6.7/3.7) after size+momentum; +36bps day-1 Q5-Q1 event-study spread (t=10).
(2) NOT tradable: with next-open execution the spread is ~0/negative; all alpha is the overnight gap after the signal close.
(3) Follow-up MOC test: the strong "enter at signal close" paper run (gross Sharpe 2.93) was LOOK-AHEAD — trade_date assigns after-hours news (~25% of articles, incl. 4-6pm earnings) to the same day. Honest version (15:45 ET knowability cutoff) has gross Sharpe -0.55. After-hours-news-only traded at next open: ~0. Conclusion: market prices this news faster than any daily-bar entry.

Review critically and bluntly:
A. Methodology bugs: look-ahead anywhere else, timezone handling (UTC->ET), trading-calendar joins, forward-return alignment, quintile construction, turnover/cost model, Newey-West implementation, survivorship.
B. Statistical validity: multiple testing, OOS discipline, sample period quirks (2020 COVID), interpretation of IC/FM magnitudes.
C. Is "real predictive content but no tradable edge at daily granularity" the right conclusion, or is there an alternative explanation (e.g., the 'predictive content' itself is contaminated by the same-day assignment of after-hours news — is the IC/FM/event-study evidence ALSO look-ahead-tainted)?
D. What cheap follow-up tests (with data already in the repo: daily OHLCV + timestamped news) would you run before fully accepting the negative result? And given the negative trading result, what NON-trading uses of the entity-sentiment model would you prioritize?

Output format: (1) one-paragraph verdict; (2) issues found, ranked by severity, with file references; (3) what you'd test next; (4) non-trading use recommendations. Be specific and technical.
