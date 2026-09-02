---
title: "The Sentiment-Head Loss: Weighted Huber + CCC"
subtitle: "Why and how the second-phase retrain fixes under-polarization"
date: "June 2026"
geometry: margin=1in
fontsize: 11pt
---

# The problem this loss solves

Evaluation showed the production model **under-polarizes**: it predicts sentiment
magnitudes about a third of the true value (predicted std $0.167$ vs. label std
$0.275$) and essentially never outputs a "very negative" score. A post-hoc
calibration test proved this is **regression attenuation** — the mathematically
correct hedge for a model with limited ranking accuracy — and the old training
recipe ($\text{MSE} + (1-\text{Pearson})$) actively encourages it. The new
`ccc_huber` loss is designed to remove that incentive.

The loss has three additive terms:

$$
L \;=\; w_{\text{huber}}\cdot \text{weightedHuber}
   \;+\; w_{\text{ccc}}\cdot \bigl(1 - \text{CCC}\bigr)
   \;+\; w_{\text{sign}}\cdot \text{signPenalty}
$$

---

# 1. Huber loss — robust regression

Huber loss blends squared error (L2/MSE) and absolute error (L1/MAE). For an error
$e = \hat{y} - y$ and threshold $\delta$:

$$
\text{Huber}_\delta(e) =
\begin{cases}
\tfrac{1}{2}\,e^{2} & \text{if } |e| \le \delta \quad(\text{MSE-like: smooth, precise near 0})\\[4pt]
\delta\bigl(|e| - \tfrac{1}{2}\delta\bigr) & \text{if } |e| > \delta \quad(\text{MAE-like: linear, gradient capped at }\delta)
\end{cases}
$$

We use $\delta = 0.1$.

**Why Huber, not MSE.** MSE squares the error, so a single badly-wrong label
produces an outsized gradient. Our labels are LLM-generated and noisiest at the
**extremes**, so under MSE one mislabeled $-0.9$ could yank the model. Huber **caps
the gradient magnitude** beyond $\delta$: large errors are still penalized, but
linearly, so noisy extremes cannot dominate. We keep MSE's precision near zero and
MAE's outlier-robustness.

**Weighted Huber** (what we actually use) scales each entity's term by

$$
w_i = \min\bigl(1 + 3\,|y_i|,\; 3\bigr).
$$

A neutral entity gets weight $\approx 1$; a strong entity ($|y_i| = 0.7$) gets
$\approx 3$. Because roughly 59% of the labels are neutral, an **unweighted** loss is
minimized by predicting $\approx 0$ everywhere (mean-collapse). Up-weighting the
strong-sentiment entities forces the model to commit to a magnitude.

---

# 2. CCC — Concordance Correlation Coefficient

CCC measures agreement with the **identity line** ($\hat{y} = y$), not merely
correlation along *some* line:

$$
\text{CCC} \;=\;
\frac{2\,\operatorname{cov}(\hat{y}, y)}
     {\operatorname{var}(\hat{y}) + \operatorname{var}(y) + \bigl(\mu_{\hat{y}} - \mu_{y}\bigr)^{2}}
\;=\;
\frac{2\,\rho\,\sigma_{\hat{y}}\,\sigma_{y}}
     {\sigma_{\hat{y}}^{2} + \sigma_{y}^{2} + (\mu_{\hat{y}} - \mu_{y})^{2}},
$$

where $\rho$ is Pearson's correlation. The denominator stacks three penalties:
prediction variance, target variance, and the squared gap between the means.

Implementation (population variance; $10^{-8}$ guards division by zero):

```python
vp = pred - pred.mean();  vy = y - y.mean()
cov = (vp * vy).mean()
ccc = 2*cov / (pred.var() + y.var() + (pred.mean() - y.mean())**2 + 1e-8)
```

**Why CCC and not Pearson — the crux.** Pearson is **scale- and
location-invariant**. A timid model predicting exactly half the true magnitude
($\hat{y} = 0.5\,y$) scores Pearson $= 1.0$ — *perfect*. Pearson literally cannot
see under-polarization. CCC **can**: shrinking the predictions shrinks
$\operatorname{cov}$ in the numerator while the denominator stays large, so CCC
falls. Maximizing CCC (i.e. minimizing $1-\text{CCC}$) forces the model to match the
labels' **spread and mean** — to stop hedging toward zero. This term directly
punishes the magnitude compression the evaluation diagnosed.

---

# 3. Sign penalty

An asymmetric hinge applied only to strong-sentiment entities ($|y_i| \ge 0.4$),
penalizing predictions on the **wrong side** of zero:

$$
\text{signPenalty} = \operatorname{mean}_{|y_i|\ge 0.4}\;
\operatorname{ReLU}\!\bigl(-\,\hat{y}_i \cdot y_i\bigr).
$$

The product $\hat{y}_i \cdot y_i$ is negative exactly when prediction and label
disagree in sign; $\text{ReLU}$ keeps only those cases. Getting the *direction*
wrong on a lawsuit or an earnings beat is worse than missing its magnitude, so this
term guards the cases that matter most.

---

# 4. How the terms work together

| term | scope | what it guarantees | failure mode if used alone |
|---|---|---|---|
| weighted Huber | **pointwise** (each $\hat{y}_i$ vs. its $y_i$) | individual accuracy, robustly | still drifts to the mean on neutral-heavy data |
| $1-\text{CCC}$ | **distributional** (whole batch vs. labels) | correct spread + alignment | wouldn't pin individual predictions |
| sign penalty | **strong entities** | no direction flips where it counts | ignores magnitude entirely |

The two main terms are complementary: Huber keeps individual predictions accurate,
while CCC keeps the overall distribution from collapsing toward neutral. Contrast the
**old** recipe, $\text{MSE} + (1-\text{Pearson})$: a mean-collapsing term plus a
scale-*blind* term — precisely why the current model is timid.

---

# 5. Empirical confirmation

Retraining only the sentiment head (frozen encoder) with this loss on the new
decisive labels moved the held-out **test** Pearson from $0.485 \to 0.643$ and the
predicted/label std ratio from $0.57 \to 1.04$ — i.e. the under-polarization is
gone — while sign-flips *fell* ($0.215 \to 0.142$) and the extreme-bucket F1 scores
rose from $\approx 0$. The loss redesign did what the diagnosis predicted.
