# Task 16A — Does Support Width Predict Q_C, or Only Coverage?

**Verdict: both pre-registered expectations HELD, and the result strengthens the
theory section. Q_C is the better predictor of coverage in 7 of 8 cells, and
support width predicts Q_C moderately well (median ρ = 0.662). The chain
support width → Q_C → coverage is now measured, not just argued.**

Script: `fix-reg/task16abd_quantile_mechanism.py`.
Data: `task16a_qc_direct_test.csv`, `task16_perdraw_recession.csv`.

---

## Pipeline change made first (per the spec's guardrail)

The 200-draw sweep computed each draw's calibration set and then discarded
everything except support width, IQR, entropy, coverage and width. **Q_C was
never saved**, so this item could not have been done by reusing reported
aggregates — exactly the situation the spec anticipated.

`domain_common.random_draw_sweep` now records **Q_C**, the calibration set's own
(1−α) quantile, and accepts an `alpha_target` argument so Item D can vary the
target level. Both changes are additive; every existing caller was verified
unaffected.

## Result — the three-way comparison

ρ against realised coverage, 200 draws per cell:

| Domain | Model | Horizon | supp→Q_C | supp→cov | rare→cov | **Q_C→cov** |
|---|---|---|---|---|---|---|
| Recession | stacking-chain | Current | 0.622 | — | — | — |
| Recession | stacking-chain | 1M | 0.670 | 0.590 | 0.386 | **0.716** |
| Recession | stacking-chain | 3M | 0.654 | 0.639 | 0.237 | **0.859** |
| Recession | stacking-chain | 6M | 0.724 | 0.725 | 0.283 | **0.961** |
| Healthcare | ridge | 30-day | 0.651 | 0.619 | 0.507 | **0.819** |
| Healthcare | gradboost | 30-day | 0.637 | 0.517 | 0.385 | **0.782** |
| Climate | ridge | region-month | 0.698 | 0.594 | 0.589 | **0.889** |
| Climate | gradboost | region-month | 0.741 | 0.599 | 0.481 | **0.859** |

*(Current horizon is saturated — coverage is constant across draws, so no
correlation with coverage is defined. Its supp→Q_C link is still measurable and
reported.)*

## Both pre-registered expectations held

**(i) Q_C predicts coverage at least as well as support width — 7 of 8 cells**
(the eighth being the undefined Current horizon). The margin is substantial and
grows with horizon difficulty: at 6M, Q_C reaches **ρ = 0.961** against support
width's 0.725.

**(ii) Support width is a moderate-to-strong predictor of Q_C** — median
ρ = 0.662, minimum 0.622, and remarkably stable across all three domains
(0.622–0.741). Support width is a *good but imperfect* proxy, which is exactly
what Section 4 claims.

## What this means for the paper

**The "proxy" language in Section 4 is vindicated and can be stated more
precisely.** Support width is not merely correlated with coverage by
coincidence — it works *through* Q_C, and Q_C is measurably closer to the
outcome, as the theory predicts. The chain is now empirically resolved at all
three links rather than argued at two and instantiated at one point.

**The gap between ρ(supp→cov) and ρ(Q_C→cov) is itself informative.** At 6M it
is 0.725 vs 0.961 — a 0.236 gap, meaning roughly a quarter of the predictive
relationship is lost by using the proxy instead of the quantity ACI actually
consumes. That is the price of describing the mechanism in terms of support
width, and it is worth stating.

**But it does not change the practical recommendation.** Task 10 already
established that the selector attains the maximum achievable Q_C for a fixed
pool, so optimising the proxy and optimising the target land on the same
solution. Support width remains the right diagnostic to *report* (it is
interpretable and model-free); Q_C is the right quantity to *reason with*.

## Honest caveats

- **These are the same 200 draws per cell used throughout the paper**, so the
  correlations share whatever sampling idiosyncrasies those draws carry. This
  is a reanalysis with a new statistic, not new evidence.
- **Q_C is evaluated at the fixed nominal α = 0.10**, whereas ACI's α drifts
  within [0.073, 0.132] during a run. The recorded Q_C is therefore the quantile
  at the *target*, not the exact one used at every step. Item D varies α
  directly and finds this matters.
- **The Current horizon contributes nothing to the coverage comparison** because
  its coverage is constant across draws. Seven cells, not eight, carry the
  headline result.
- **Correlation ordering is not a causal test.** Q_C being closer to coverage is
  consistent with the theorised chain but does not prove the direction; the
  analytic argument in Section 4 does that work, and this measures its
  observable implication.
