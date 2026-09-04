# CLAUDE.md — working notes for agents in this repo

## What this repo is

Public GitHub repo (`Russell-Ding/entity_sentiment_model_pipeline`) for Russell's
independent project: a Longformer entity-sentiment model for financial news, plus the
`signal_strategy/` study testing whether the scores predict S&P 500 returns.
`docs/project_report.pdf` (source `project_report.tex`) is the public write-up and
`README.md` is the front door; keep both accurate when results or layout change.

## Public-release layout — 2026-09-03

The public tree holds only: the report, the model code (`models/`), the label-schema /
batching / metrics part of `training/`, the evaluation, inference and postprocessing
scripts, `scripts/analysis/inspect_ner_head.py`, the signal study, model cards + metrics,
the labelling spec and prompts, `data/reference/*.csv`, and the inference/evaluation
notebooks.

Everything else stays **on disk (in the Drive folder) but untracked and gitignored** — see
the "Not part of the public release" block in `.gitignore`: training scripts and
notebooks, `training/trainer.py`, labelling / collection / preprocessing scripts,
`data_collection/`, `config/`, the retrain guides and loss explainer in `docs/`, source
tiering outputs, training curves, `data/DATASET_README.md`. The older working-file archive
is `_archive_local/` (also ignored). The private technical notes (the three-part series:
model structure / training / signal analysis) and their external reviews live in
`docs/private/` (ignored).

Rules:
- Do not re-add any excluded file without asking Russell. If a kept script needs something
  from an excluded file, move that piece into a kept module (as `training/metrics.py` was
  split out of `trainer.py`) rather than re-adding the file.
- The local Colab training workflow still imports `training.trainer`; keep
  `training/__init__.py`'s guarded import so the package works with and without it.
- The report's appendix prompts are ASCII copies in `docs/report_assets/`, generated from
  `data_label_criteria/`; regenerate them if the prompts change. Build the report from
  `docs/` with `pdflatex project_report.tex` (three passes).

## History

- 2026-09-02: local `master` squashed to a single orphan commit (old history contained
  personal paths); the pre-squash history is preserved locally as
  `archive/pre-public-history` — never push that branch.
- 2026-09-03: public-release trim (above), report added, corrections applied (below).
  At that point the GitHub remote still held the old 10-commit history and the repo was
  private; publishing means `git push --force origin master` and flipping visibility.

## Corrections applied 2026-09-03 (verified — do not undo)

- The `signal_strategy/` scores came from the **v2.0** checkpoint, not v2.1: inference
  files are dated 24 May – 7 Jun 2026 (v2.1 packaged 20 Jun), `inference_launcher.ipynb`
  hard-codes v2.0, and re-scoring stored articles matches v2.0 exactly. Correction notes
  sit at the top of `RESULTS.md`, `CONSOLIDATED_REPORT.md` and `signal_strategy/README.md`;
  any new write-up must attribute the scores to v2.0 unless inference is re-run.
- The v2.1 sentiment head was warm-started from the v2.0 head (`train_sentiment_arm.py`
  loads the full state dict; ledger baseline CCC 0.4147 = v2.0 head). The model card now
  says so.
- The CRF transition/start/end scores in v2.0 and v2.1 never trained (all within ±0.11;
  no BIO constraints; Viterbi = argmax on >99% of tokens) because `train_two_stage.py`
  builds the model without `ner_label_to_id`. Fix at the next retrain: pass the label map
  at construction and give CRF parameters their own LR.

## Ground rules

- Root holds only folders, `README.md`, `CLAUDE.md`, `LICENSE`, `requirements.txt`,
  `.gitignore`. Scripts go under `scripts/<stage>/`, docs under `docs/`, generated files
  under `outputs/` or `_archive_local/`.
- Data (`data/raw`, `data/labeled`), checkpoints, weights, inference dumps and logs are
  never committed — EODHD data is not redistributable and the weights are ~1.8 GB.
- Secrets only in `config/secrets.yaml` (ignored) or env vars.
- Point-in-time discipline in `signal_strategy/`: any new signal test must use the
  15:45 ET knowability cutoff (`moc_feasibility.py`) — the naïve `trade_date` panel is
  known to be contaminated (see `CONSOLIDATED_REPORT.md` §5).
- Never commit an absolute local path or an email address. Before any push: grep the
  tracked files for the macOS users prefix, `gmail` and `sk-ant-` — excluding this file, it
  must return nothing.
- Commit and push only when Russell asks.

## Open items

- Commit a `feasible_analysis.py` that rebuilds `panel_feasible.parquet` and re-runs IC,
  Fama-MacBeth and the event study on it (the correction is currently only IC + backtest).
- Test pre-market news (05:00–09:00 ET, ~28% of articles, knowable before the 9:30 open) traded
  at the same day's open.
- Check the 2023–24 negative-IC reversal hint.
- Point-in-time S&P 500 membership (current universe = survivorship bias).
- Re-score the tier-1 corpus with v2.1 and re-run the study.
- Private notes parts 2 (training) and 3 (signal analysis) in `docs/private/`.
