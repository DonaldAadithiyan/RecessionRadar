# Task 16B — Multivariate Decomposition, N Held Fixed

**Verdict: support width dominates in every cell, but rare-count is NOT
redundant everywhere. Its confidence interval excludes zero in 5 of 7 cells,
with partial R² up to 0.117. This is a more nuanced answer than the
within-tertile check gave, and the paper should adopt it.**

Script: `fix-reg/task16abd_quantile_mechanism.py`.
Data: `task16b_multivariate.csv`.

Model, per cell, on the 200 draws (N fixed at its published value throughout —
the draws already hold it fixed, so this is a direct extension):

```
coverage ~ β₀ + β₁·SupportWidth + β₂·RareCount + ε      (standardized)
```

---

## Result

| Domain | Model | Horizon | β(supp) | 95% CI | partial R² | β(rare) | 95% CI | partial R² | rare CI excl. 0 |
|---|---|---|---|---|---|---|---|---|---|
| Recession | chain | 1M | **+0.548** | [.42, .68] | 0.290 | +0.170 | [.04, .30] | 0.038 | **yes** |
| Recession | chain | 3M | **+0.633** | [.51, .76] | 0.385 | +0.050 | [−.07, .17] | 0.004 | no |
| Recession | chain | 6M | **+0.706** | [.60, .81] | 0.492 | +0.074 | [−.03, .18] | 0.011 | no |
| Healthcare | ridge | 30-day | **+0.469** | [.33, .61] | 0.214 | +0.291 | [.15, .43] | 0.095 | **yes** |
| Healthcare | gradboost | 30-day | **+0.419** | [.27, .57] | 0.151 | +0.203 | [.06, .35] | 0.040 | **yes** |
| Climate | ridge | region-month | +0.329 | [.19, .47] | 0.095 | **+0.368** | [.23, .51] | 0.117 | **yes** |
| Climate | gradboost | region-month | **+0.466** | [.32, .61] | 0.181 | +0.203 | [.06, .35] | 0.040 | **yes** |

*(Current horizon excluded: coverage is constant across draws, so there is no
variance to decompose. Reported as a skip rather than a spurious regression.)*

## What this changes

**The recession testbed behaves exactly as the paper claims.** At 3M and 6M —
the horizons with real coverage variance — rare-count's CI **includes zero** and
its partial R² is 0.004 and 0.011. Support width carries 0.385 and 0.492. That
is genuine redundancy, and it is the strongest version of the paper's claim.

**But the other domains do not.** Rare-count's CI excludes zero in both
healthcare cells and both climate cells, with partial R² of 0.040–0.117. In
**climate/ridge it is the larger of the two coefficients** (β = 0.368 vs 0.329).

This is more honest and more informative than the within-tertile check, which
averaged over tertiles and reported a single attenuated ρ. The regression
separates the two questions the check conflated: *does rare-count still predict
coverage once support width is controlled* (yes, in 5 of 7 cells), and *is it
the dominant predictor* (no, except climate/ridge).

**The paper's claim should therefore be stated as it was already scoped in Task
1**: diversity **dominates** rare-count, not that rare-count is irrelevant. The
near-total redundancy is a US-recession-at-long-horizons result. This analysis
now quantifies that distinction with confidence intervals rather than asserting
it.

## Per the guardrail: near-zero partial R² reported, not omitted

The spec required reporting rare-count's partial R² **even when approximately
zero**, since that is confirmatory. Recession 3M (0.004) and 6M (0.011) are
those cells, and they are the paper's headline horizons — the confirmation lands
exactly where the claim is strongest.

## Honest caveats

- **Linear and additive.** The model assumes coverage responds linearly to both
  standardized predictors with no interaction. The underlying relationship
  (quantile reach → coverage) is a step function in a finite test stream, so
  this is an approximation; it is the same approximation every correlation in
  the paper already makes.
- **Support width and rare-count are correlated** (Task 15 measured ρ = 0.509 in
  healthcare). Collinearity inflates both standard errors, making the CIs
  conservative — which cuts against finding rare-count significant, so the 5-of-7
  result is not an artifact of it.
- **200 draws per cell**, the same draws used throughout. The CIs describe
  sampling variation across those draws, not across datasets or time.
- **Model R² is not reported as a headline** because the two predictors are
  proxies for Q_C (Task 16A), not causes; the decomposition answers a relative
  question, not "how much of coverage is explained".
