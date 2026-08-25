# Task 31, Item 1 — Domain Screening for Concept Drift

## Result: **epidemic (OWID COVID) qualifies; fraud is rejected.** Both screened with `task30_1_shift.py`'s diagnostic applied unchanged, before either touched the method.

Data: `task31_1_screening.csv` · Build: `task31_build.py` · Screen: `task31_1_screen.py`

## The bar, set from Task 30's own numbers (not lowered)

A candidate qualifies only on genuine drift in **P(error | z) at matched z-bins**:
`frac_zbins_error_drift ≥ 0.20` **and** median error ratio outside [0.8, 1.25].
Covariate shift alone does not qualify — that is the climate pattern.

## Every candidate screened

| candidate | classifier AUC (P(X) shift) | **drifted z-bins** | median err ratio | qualifies |
|---|---|---|---|---|
| **fraud** (credit-card) | 0.477 | **0.00** | 1.047 | **No** |
| **epidemic** (OWID COVID) | 0.992 | **1.00** | **0.443** | **Yes** |
| *energy (reference, qualified)* | *0.799* | *0.40* | *1.315* | *Yes* |
| *climate (reference, control)* | *0.958* | *0.00* | *1.276* | *No* |

## Fraud — rejected, and the rejection is informative

| bin | median &#124;e&#124; FIT | median &#124;e&#124; TEST | KS p |
|---|---|---|---|
| 1 | 0.573 | 0.826 | 0.057 |
| 2 | 0.632 | 0.704 | 0.557 |
| 3 | 0.807 | 0.678 | 0.284 |
| 4 | 1.065 | 0.882 | 0.137 |
| 5 | 1.497 | 1.568 | 0.563 |

Zero bins drift, and the classifier AUC of **0.477** — below chance — shows there
is no detectable P(X) shift either.

**The reason is a known defect in this data source, recorded honestly:** the
OpenML copy of the credit-card fraud dataset has **no `Time` column**. I used row
order as a proxy for arrival order and flagged in advance that if the proxy were
wrong the screen would be invalid. An AUC of 0.477 says the "early" and "late"
halves are statistically indistinguishable — i.e. the rows are **not** in time
order, so the proxy failed. Fraud is therefore rejected not because adversarial
concept drift is absent in fraud generally (it is textbook), but because **this
dataset as distributed cannot express it**. A properly timestamped fraud feed
would still be a good candidate for a future task.

## Epidemic — qualifies decisively

| bin | n_fit | n_test | median &#124;e&#124; FIT | median &#124;e&#124; TEST | KS p |
|---|---|---|---|---|---|
| 1 | 347 | 180 | 0.187 | 0.170 | **0.0000** |
| 2 | 347 | 113 | 0.312 | 0.196 | **0.0006** |
| 3 | 346 | 67 | 0.527 | 0.136 | **0.0000** |
| 4 | 347 | 34 | 1.191 | 0.275 | **0.0000** |

**4 of 4 bins drift at p<0.001** — the strongest concept-drift signal of any
domain in this project (energy managed 0.40). AUC 0.992 confirms large covariate
shift as well, so this is compound shift like energy's.

Target: national daily COVID deaths per million; predictors are strictly lagged
(7/14/21-day) case, positivity, ICU, hospitalization, stringency and vaccination
series. 8,664 usable country-days; most recent 4,400 used.

## An important asymmetry, flagged before Item 3 runs

**The drift runs in the opposite direction to energy's.** Energy's errors grew
(ratio 1.315, bin 5 at 2.25×); epidemic's errors **shrink** (ratio 0.443, bin 4
falling 1.191 → 0.275). The late-pandemic test period is *easier* to predict than
the fit period at matched difficulty — variants, vaccination and accumulated
immunity flattened the case→death relationship.

This makes epidemic a **harder and better** test of claim 3 than a
same-direction domain would be:

- Energy's result could be explained trivially — any method that widens under
  observed misses does better when errors grow. Online adaptation is *rewarded*
  by growing errors regardless of mechanism.
- Epidemic inverts that. A frozen calibration-period estimate will now be **too
  wide**, over-covering with wasteful intervals, and an online-adapting method
  must *narrow* to win. The mechanism claim ("online anchor adapts, frozen
  estimate does not") predicts errordir wins in **both** directions; a
  reward-for-widening artifact predicts it wins only in energy's.

If claim 3 confirms here it is materially stronger than an n=2 count suggests,
because the two domains stress the mechanism in opposite directions.

## Selection

**Epidemic proceeds to Items 2–3.** Fraud is reported and dropped, per the
guardrail that all screened candidates be reported.
