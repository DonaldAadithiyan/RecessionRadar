# Task 24, Items 2–3 — Four Independent Fits and the Fresh Validation Gate

**Result: 2 of 4 validated — `healthcare/ridge` and `climate/ridge`. Both
gradboost fits fail Check 2, for a reason that is a property of the check
applied to tree models, not evidence that β is worse there.**

Script: `task24_2_beta_fits.py` · Data: `task24_3_validation.csv`

## The four fits

Each is fresh: its own domain's pool, its own 60/40 FIT/CAL split (asserted
disjoint), its own RidgeCV regression of |error| on its own features.
Recession's β is never transferred, and healthcare and climate are never pooled.

| domain/model | n_fit | n_cal | features | fit r | **held-out CAL r** | disag weight |
|---|---|---|---|---|---|---|
| healthcare/ridge | 190 | 175 | 21 | 0.512 | 0.149 | 6.8% |
| healthcare/gradboost | 190 | 175 | 21 | 0.502 | 0.128 | 13.0% |
| climate/ridge | 597 | 551 | 22 | 0.292 | **0.335** | 1.2% |
| climate/gradboost | 597 | 551 | 22 | 0.300 | **0.374** | 6.1% |

**Climate's β generalizes far better than healthcare's.** Climate held-out r of
0.335–0.374 against healthcare's 0.128–0.149. Climate's fit r is *lower* than
healthcare's (0.292 vs 0.512) while its held-out r is more than double —
healthcare is overfitting the FIT split, climate is not. With n_fit=597 vs 190
that is the expected direction.

For reference, recession/6M reached held-out r = 0.539 (Task 23), so climate
lands between the two domains and healthcare is clearly weakest.

## The gate (Item 3), full four-row table

| domain | model | angle PC1 | Check 1 | perturb pct | Check 2 | **validated** |
|---|---|---|---|---|---|---|
| healthcare | ridge | 89.63° | PASS | **0.980** | PASS | **YES** |
| healthcare | gradboost | 88.13° | PASS | 0.705 | FAIL | No |
| climate | ridge | 85.79° | PASS | **1.000** | PASS | **YES** |
| climate | gradboost | 79.49° | PASS | **0.070** | FAIL | No |

Check 1 passes everywhere (79–90° from PC1 — β is nowhere near the generic
top-variance direction in any domain).

## Why both gradboost fits fail — the mechanism matters

The naive reading is "β is worse for gradboost." The numbers say otherwise:

| domain/model | perturb effect (β) | null p95 |
|---|---|---|
| healthcare/ridge | 2.531 | 2.273 |
| healthcare/gradboost | 2.296 | 2.788 |
| climate/ridge | 0.850 | **0.530** |
| climate/gradboost | 0.857 | **1.604** |

**β's absolute effect is essentially identical across model classes** — climate
0.850 (ridge) vs 0.857 (gradboost). What differs is the *null*: gradboost's 95th
percentile is 3× ridge's (1.604 vs 0.530). Random directions move a boosted-tree
model far more than they move a ridge model.

The reason is structural. Ridge is linear, so perturbing along direction *v*
changes the prediction by exactly `w·v × magnitude` — smooth, and a direction
aligned with the weight vector reliably beats random ones. Gradboost is a step
function over axis-aligned splits: a perturbation either crosses splits (large
jump) or doesn't (zero change). That makes the random-direction null both
higher-variance and heavier-tailed, so clearing its 95th percentile is a much
harder bar for any single direction.

**This is a property of Task 21's Check 2 applied to non-smooth models, not a
verdict on β.** But the gate is defined as it is, and per the guardrail ("no fit
gets to skip this gate"), the failing fits do not proceed. Loosening the
threshold for tree models after seeing that they fail would be exactly the kind
of post-hoc adjustment this project's discipline forbids.

Recording it explicitly matters for interpretation: the honest statement is that
Check 2 **cannot currently discriminate** for gradboost in these domains, not
that gradboost's β was tested and found wanting. A future task wanting to test
tree models properly would need a perturbation check designed for non-smooth
predictors — a scoped change, not a threshold tweak.

## Note on the disagreement feature

Per Item 1, healthcare/climate get one disagreement feature (`|pred_ridge −
pred_gb|`) where recession got three. β puts little weight on it in the
validated fits — 6.8% (healthcare/ridge) and **1.2%** (climate/ridge) — versus
recession/6M's 11.5%. Notably the two *failing* fits put the most weight on it
(13.0%, 6.1%), echoing Task 23's finding that heavy disagreement loading and
Check 2 failure travel together. With n=4 that is a pattern worth noting, not a
result.

Climate/ridge validating on a 1.2% disagreement weight means climate's signal is
essentially **all base features** — the disagreement analog, though legitimate,
contributes almost nothing there.

## Proceeding to Item 4

Validated fits: `healthcare/ridge`, `climate/ridge`.
