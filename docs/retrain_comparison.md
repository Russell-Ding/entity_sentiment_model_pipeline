# Retrain ablation — side-by-side

Baseline = current v2.0; floor = + affine calibration (Phase 0). Arms eval on the held-out **test** split. Improvement = Pearson/CCC up; under-polarization fixed = std_ratio→1 & extreme F1>0 **without** worse sign_flip/over_neutral.

| source | N | Pearson | Spearman | CCC | std_ratio | MAE | over_neut | sign_flip | vneg_F1 | vpos_F1 | ORG_r | TICKER_r | PERSON_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2.0 current (full set) | 164,723 | 0.485 | 0.451 | 0.416 | 0.568 | 0.209 | 0.308 | 0.215 | 0.00 | 0.01 | 0.486 | 0.622 | 0.493 |
| v2.0 + affine calib (floor) | 82,628 | 0.483 | 0.450 | — | 0.488 | 0.210 | 0.354 | 0.204 | 0.00 | 0.00 | — | — | — |
| armC__v1 (test) | 7,780 | 0.690 | 0.654 | 0.686 | 1.099 | 0.193 | 0.154 | 0.134 | 0.15 | 0.38 | 0.677 | 0.853 | 0.716 |
| control (test) | 8,078 | 0.643 | 0.600 | 0.641 | 1.033 | 0.196 | 0.213 | 0.151 | 0.13 | 0.37 | 0.627 | 0.807 | 0.678 |
