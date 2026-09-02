# CLAUDE.md — working notes for agents in this repo

## What this repo is

Public GitHub repo (`Russell-Ding/entity_sentiment_model_pipeline`) for Russell's
independent project: a Longformer entity-sentiment model for financial news, plus the
`signal_strategy/` study testing whether the scores predict S&P 500 returns.
`README.md` is the front door; keep it accurate when results or layout change.

## Repo reorganisation — 2026-09-02 (Cowork session)

Applied to make the repo public. Do not undo without asking Russell.

1. **Local archive.** Everything that was noise for a public reader was *moved*, not
   deleted, into `_archive_local/` (gitignored, stays on disk):
   `write-up/` (294 batch-labeling reports, old diagrams, Jan blog draft),
   `scripts/archive_20260524/`, `scripts/archived_20260116/`, `docs/labeling_reports/`,
   `data_label_criteria/archive/`, `CLAUDE (1).md` (the old guidelines), `AGENTS.md`,
   `to-do/`, `validation_set_quality_assessment.md`, `news_api_tests/`, `.codex/`,
   `outputs/{eodhd_duplicate_flags.json, eodhd_probe_results.json, source_classification_*.jsonl,
   retrain_armC__v1*, retrain_control*, e2e_evaluation/local}`,
   `signal_strategy/notes/claude_code_brief.md`, `signal_strategy/notes/reviews/*_err.txt`,
   and a copy of the old README as `README_old_jan2026.md`.
2. **Docs consolidated into `docs/`:** `financial_entity_sentiment_architecture.md` →
   `docs/architecture_spec.md`; `RETRAIN_GUIDE.md` → `docs/RETRAIN_GUIDE.md`;
   `outputs/retrain_comparison.md` → `docs/retrain_comparison.md`;
   `write-up/global_attn_dropout_proposal.pptx` → `docs/`.
3. **Personal-path scrub.** All absolute home-directory paths (macOS `~` expanded) and machine-specific Colab
   Drive paths in tracked files were replaced: shell scripts
   now use `PROJECT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"`,
   `plot_nvda_timeseries.py` uses `Path(__file__).resolve().parents[2]`, metric JSON/TXT
   use the placeholder `<repo>`, and notebooks use `/content/drive/MyDrive/entity_sentiment_model_pipeline`.
   **Rule going forward: never commit an absolute local path or an email address.**
4. **Notebook outputs stripped** (all `notebooks/*.ipynb`; sizes fell from up to 576 KB to ≤56 KB).
5. **`.gitignore` rewritten:** adds `_archive_local/`, `*.parquet`, `*.jsonl.gz`,
   `outputs/inference/`, `*_predictions_*.jsonl`, tokenizer dirs; keeps
   `trained_model/**/{MODEL_CARD.md,config.json,evaluation_results*.json}`,
   `data/reference/*.csv`, `signal_strategy/outputs/ic_daily.csv`.
   `outputs/retrain_runs.jsonl` and `data/collection_tracking.json` were un-tracked.
6. **Added:** `LICENSE` (MIT, code/docs only — data and weights excluded), new two-part
   `README.md`, author's notes at the top of `signal_strategy/RESULTS.md` and
   `signal_strategy/CONSOLIDATED_REPORT.md`, this file.
7. **History squashed** to a single orphan commit before force-pushing `master`
   (old history contained personal paths). The pre-squash history is preserved locally
   as the branch `archive/pre-public-history` — never push that branch.

## Ground rules (carried over from the old CLAUDE (1).md, condensed)

- Root holds only folders, `README.md`, `CLAUDE.md`, `LICENSE`, `requirements.txt`,
  `.gitignore`. Scripts go under `scripts/<stage>/`, docs under `docs/`, generated files
  under `outputs/` or `_archive_local/`.
- Data (`data/raw`, `data/labeled`), checkpoints, weights, inference dumps and logs are
  never committed — EODHD data is not redistributable and the weights are ~1.8 GB.
- Secrets only in `config/secrets.yaml` (ignored) or env vars; the template is the only
  key-shaped file allowed in git.
- Point-in-time discipline in `signal_strategy/`: any new signal test must use the
  15:45 ET knowability cutoff (`moc_feasibility.py`) — the naïve `trade_date` panel is
  known to be contaminated (see `CONSOLIDATED_REPORT.md` §5).
- Before any push: grep the tracked files for absolute home paths and email addresses
  (pattern: the macOS users prefix, `gmail`, `sk-ant-`) — excluding this file, it must return nothing.

## Open items (from the external reviews)

- Commit a `feasible_analysis.py` that rebuilds `panel_feasible.parquet` and re-runs IC,
  Fama-MacBeth and the event study on it (the correction is currently only IC + backtest).
- Test pre-market news (05:00–09:30 ET, ~28% of articles) traded at the same day's open.
- Check the 2023–24 negative-IC reversal hint.
- Point-in-time S&P 500 membership (current universe = survivorship bias).
