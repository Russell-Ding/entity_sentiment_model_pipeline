# Second-phase sentiment-head retrain — plan (training-process changes)

## Context / why
Evaluating the current production model (v2.0) on the new decisive DeepSeek+Kimi
labels (`labels_10ticker.final.jsonl`, 30,829 articles) showed:
- e2e Pearson r **0.485** / teacher-forced r **0.464** vs **0.628** on the Sonnet holdout.
- **Teacher-forced ≈ e2e → the gap is the SENTIMENT HEAD, not NER** (gold spans don't
  help; matched-span IoU 0.968 is clean).
- Mechanism = **magnitude under-polarization**: pred std **0.167** vs gold **0.275**;
  the model essentially never predicts `very_negative` (bucket col empty) and rarely
  `very_positive`; 30.8% of |gold|≥0.4 entities predicted ~flat; 21.5% sign flips.

Root cause in code (not just data):
- `training/trainer.py:136` — sentiment loss is **pure masked MSE**
  (`nn.MSELoss(reduction="none")`). On a ~59%-neutral label distribution, MSE's
  optimum under uncertainty is the conditional mean → it rewards shrinking toward 0.
- `models/sentiment_head.py:92` — output bias zero-init "Start predicting ~0".
- The v2 Stage-3 variant added `(1 − Pearson r)`, but Pearson is **scale-invariant**,
  so it improves ranking yet does nothing to penalize magnitude compression.

Conclusion: retraining head-only on the new (more decisive) labels will help, but
**will not fully fix under-polarization unless the loss/recipe changes.** This plan
updates the training process and verifies the change with a controlled ablation.

## Goals (success criteria)
Primary (in-distribution test split of the new set):
- Pearson r **> 0.60** AND Spearman ρ **> 0.58** (up from 0.46–0.49).
- **pred_std ≥ 0.85 × target_std** (the compression metric — the real fix).
- Extreme buckets (`very_neg`/`very_pos`): **precision/F1**, not just recall > 0
  (a model can fake recall by hallucinating extremes — Codex). Sign-flip rate < 12%.
- **Neutral false-polarization** kept low: `%(|gold|<0.1 predicted |pred|>0.3)`.
Secondary (cross-distribution): report on the Sonnet holdout (do not optimize for it).

## Phase 0 — Post-hoc calibration baseline (do this FIRST; may avoid the retrain)
Before any training, fit a trivial calibrator on the **current** model's predictions:
- affine `pred' = a·pred + b` (least-squares on val) and/or **isotonic regression**.
- Re-score the in-distribution test with the calibrator; recompute Pearson/Spearman,
  std-ratio, bucket precision/recall, sign-flips.
If a simple rescale recovers most of the lost dispersion and bucket recall, the deficit
is largely a **calibration-slope** problem, not representational — the retrain must then
**beat this calibrated baseline** to justify itself (and may be unnecessary). This is the
cheapest possible test and gates the rest of the plan. (Codex: "the diagnosis sounds
partly like a calibration/slope problem.")

### Phase 0 RESULT (run 20260619, `calibration_baseline.json`) — calibration does NOT fix it
`scripts/evaluation/calibration_baseline.py` on the 164,723 saved e2e predictions
(fit/eval split by article id). MSE-optimal affine fit = **`pred' = 0.858·pred − 0.009`
(slope < 1 — the optimal calibrator SHRINKS, not expands).** All metrics ~unchanged:
Pearson 0.483→0.483, Spearman 0.450, std_ratio 0.57→0.49, MAE 0.210, very_neg/very_pos
recall ≈ 0 throughout.

**Conclusion:** under-polarization is **regression attenuation** — the mathematically
correct hedge for a model with only r≈0.48 discrimination, NOT a miscalibrated slope.
Codex's "calibration slope" hypothesis is **refuted**. Consequences for the plan:
- The retrain is justified on **correlation/discrimination** grounds (no cheap rescale
  closes 0.48→0.63).
- **Loss tweaks alone (std-gap, magnitude-weighting) are insufficient and risk harm** —
  forcing std_ratio→1 (slope≈1.76) on unchanged ranking would add sign-flips & MAE.
  CCC remains the right objective, but priority shifts from "decisiveness" to "separability."
- **Head-only is unlikely to suffice.** If the frozen encoder caps correlation at ~0.48,
  no head/loss change exceeds it. **Promote Arm C (unfreeze top 1–2 encoder layers) to a
  PRIMARY arm**, not a fallback. Diagnostic priority: does unfreezing lift Pearson?

## Training-process changes (the core of this plan)

### 1. Loss redesign — directly penalize magnitude compression
Replace pure MSE. **Primary candidate (per Codex):**

    L = weightedHuber(pred, y, w=clip(1 + α·|y|, max≈3))
        + λ_ccc · (1 − CCC(pred, y))
        + (optional) λ_sign · signFlipPenalty for |y| ≥ 0.4

- **CCC (concordance correlation)** replaces `(1 − Pearson)`: it penalizes correlation,
  **mean shift, AND scale mismatch** in one term — directly targets ranking + calibration
  + magnitude. Pearson is scale-invariant and was the gap; CCC is not.
- **Weighted Huber** (not MSE): robust to noisy LLM extremes while still rewarding
  magnitude. Cap the magnitude weight at ~3 so noisy strong labels don't dominate.
- **Asymmetric sign-flip penalty** on |y| ≥ 0.4 (getting a lawsuit's sign wrong is worse
  than a magnitude miss).
- **std-gap is DEMOTED to optional** (Codex: noisy/gameable at batch level). Only add if
  CCC + weighting still under-disperse; if used, form is `(log(std_pred/std_gold))²` over
  a gradient-accumulated/large effective batch, not absolute per-batch difference.

### 2. Counter the neutral majority
The new set is ~59% neutral. Magnitude-weighting (above) is the primary lever; as an
alternative/ablation, per-batch downsampling of |target|<0.1 entities. Avoid destroying
calibration — we still need genuine 0.0 (incidental) predictions, so prefer
re-weighting over hard dropping.

### 3. Keep encoder + NER frozen; train sentiment head only (as before)
Ranking signal (r≈0.5) shows the representation already carries the signal — the head
mapping is the bottleneck. Start head-only (cheap, ~530K params). **Fallback:** if the
new loss plateaus below target, unfreeze the **top 1–2 encoder layers** with a low LR.

### 4. Output head
`0.95·tanh` cap is adequate (targets ±1; atanh(0.9/0.95)≈1.9 is easily reachable), so
the cap is not the binding constraint — the loss is. Keep as-is; optionally widen to
`1.0·tanh`. Keep zero-bias init.

### 5. Validation / early-stopping metric change
Do **not** early-stop on MSE (it rewards timidity). Select on **CCC** (or a composite).
Log (monitor, don't all optimize): Pearson, Spearman, **CCC**, pred/gold **std ratio**,
**mean bias**, MAE by magnitude bucket, **sign-flip rate by bucket**, **extreme-bucket
precision/recall/F1**, **neutral false-polarization** `%(|gold|<0.1 & |pred|>0.3)`,
and **tanh saturation rate** (if many outputs near the ±0.95 cap, calibration is broken).
Also: undertraining mimics compression — ensure enough epochs / a proper LR schedule.

### 6. Data splits
- Split the new set **by article id** (no entity leakage) → train/val/test ≈ 90/5/5.
- **Near-duplicate dedup across splits** (Bloomberg reruns leak even with distinct ids).
- Add a **date/event-based OOD holdout** (by-id alone doesn't prevent same-event leakage).
- Primary metric = in-distribution **test** (untouched until final selection; tune loss
  weights on **val** only via a small grid). Sonnet holdout = OOD cross-check.
- **Article-id grouped bootstrap** for all CIs (entity-row bootstrap inflates confidence).
- Keep the two label sources **separate**; include Kimi-rescored persons (cleaner).

## Controlled ablation (attribute the gain) — expanded per Codex
From the same v2.0 checkpoint, identical splits/optimizer/epochs/early-stop budget,
**multiple seeds**, with article-id grouped bootstrap CIs:
- **Baseline:** current model + Phase-0 affine/isotonic calibration (the bar to beat).
- **Arm A:** OLD recipe (plain MSE [+ (1−Pearson)]) on new labels — data effect.
- **Arm B:** NEW loss (weighted-Huber + CCC) — recipe effect.
- **Arm D:** magnitude-weighting only (isolate from CCC).
- **Arm E:** CCC + weighted-Huber, no std-gap (isolate std-gap's contribution).
- **Arm C:** Arm B + unfreeze top 1–2 encoder layers (low encoder LR 1e-6–3e-6,
  discriminative head LR) — tests whether the head-only ceiling is representational.
Diagnostic read: head-only improves std but not rank → unfreeze (Arm C); rank improves
but still compressed → loss/calibration. All arms must beat the Phase-0 calibrated
baseline to justify retraining.

## Files — BUILT (harness ready; runs on Colab A100)
- `scripts/data/split_new_dataset.py` — by-id train/val/test + near-dup dedup. RAN:
  train 27,015 / val 1,558 / test 1,558 (698 near-dups dropped), splits disjoint.
- `models/sentiment_head.py::configure_loss` + `compute_loss` — configurable recipe;
  default unchanged (= Arm A legacy MSE+(1−Pearson)); `ccc_huber` mode adds
  weighted-Huber + (1−CCC) + asymmetric sign penalty. (std-gap intentionally omitted
  per Phase-0: discrimination, not dispersion-forcing, is the lever.)
- `training/trainer.py::compute_sentiment_metrics` — added CCC, std_ratio, sign_flip,
  neutral_false_polar, over_neutral, very_neg/very_pos F1. (Tested locally.)
- `notebooks/retrain_ablation_launcher.ipynb` — arm-parameterized (armA/control/armC/
  armD/armE); loads v2.0, per-arm freeze + Arm-C top-2-layer unfreeze with
  discriminative LR, selects best by **CCC**, evals on TEST with e2e + analyze_eval_gaps
  + calibration floor.
- Eval reuse: `evaluate_e2e_pipeline.py`, `analyze_eval_gaps.py`,
  `calibration_baseline.py`.

## Verification
- Ablation Arm B beats Arm A and the 0.46–0.49 baseline on in-distribution test.
- pred_std/target_std ≥ 0.85; very_neg/very_pos bucket recall materially > 0.
- Calibration curve (analyze_eval_gaps.py) lands near the diagonal vs the current
  compressed curve.
- Sonnet-holdout cross-check doesn't collapse (sanity that we didn't overfit decisiveness).

## Label-noise guardrails (Codex: the biggest risk)
"More decisive" ≠ "more correct." Optimizing decisiveness can overfit teacher style.
- Slice metrics by **label source/version**, entity type, ticker, source, date.
- **Manually audit high-confidence model↔label disagreements** (are they model errors
  or label errors?).
- Calibration curves **per entity type and per magnitude bucket**.
- Where Sonnet AND DeepSeek/Kimi labeled comparable cases, check agreement.
- Person sentiment is least stable — watch it specifically.
- Gate: do NOT reward extreme-bucket recall without extreme-bucket **precision**.

## Reviewer status
- **Codex review: incorporated** (this revision). Key changes adopted: Phase-0
  calibration baseline as a gate; CCC + weighted-Huber as primary loss (std-gap demoted);
  capped magnitude weight; extreme-bucket precision + neutral-false-polarization metrics;
  Arms C/D/E + seeds + grouped-bootstrap CIs; dedup + date/event OOD holdout; label-noise
  audit. Codex verdict: "directionally good… demote batch std-gap, add CCC/calibration
  baselines, include a top-layer-unfreeze arm, make decisiveness subordinate to
  calibrated correctness."
- Optional: second opinion from Kimi (not yet run).
