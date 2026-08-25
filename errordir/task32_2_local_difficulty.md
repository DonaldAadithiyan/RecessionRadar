# Task 32, Item 2 — Candidate A: Local kNN Difficulty

**Result: validates on both ridge arms (2/2), fails on all four tree arms (0/4).**

Script: `task32_2_local_difficulty.py` · Data: `task32_4_validation.csv`

## Construction

`difficulty(x)` = mean OOF |error| of x's K nearest FIT-period neighbours in
standardized feature space. **No global direction** — the signal is defined
pointwise from local history, so it cannot make the smooth-consistent-gradient
assumption that broke β on trees. It satisfies the task's design bar.

`K = max(20, ceil(n_fit/50))`, fixed in advance: the floor of 20 so the local
error mean rests on ≥20 observations (same reasoning as Task 22's p99 floor and
Task 28's window floor); `n_fit/50` gives ~50 effective local regions. No
test-period tuning of K or the distance metric.

## Results

| domain | model | class | held-out r | angle PC1 | perturb pct | validated |
|---|---|---|---|---|---|---|
| insurance | xgboost | tree | 0.021 | 64.6° | 0.515 | **No** |
| insurance | lightgbm | tree | 0.031 | 65.0° | 0.395 | **No** |
| insurance | **ridge** | linear | 0.022 | 75.3° | **0.965** | **Yes** |
| energy | xgboost | tree | 0.371 | 72.5° | 0.740 | **No** |
| energy | lightgbm | tree | 0.414 | 72.6° | 0.775 | **No** |
| energy | **ridge** | linear | 0.433 | 77.1° | **1.000** | **Yes** |

Check 1 passes everywhere (64–77° from PC1). Check 2 fails on every tree arm.

## The ridge control did its job

This is the arm that makes the result interpretable. Candidate A **validates on
both ridge fits** using the same data, same features, same K, same procedure.
So the domains contain a usable signal and the mechanism itself works — the
failure is specific to tree model class, not to the data or to Candidate A's
construction.

Note energy/ridge reaches perturbation percentile **1.000** with held-out
r=0.433, essentially matching β's own performance there. Candidate A is a
perfectly good difficulty signal; it just cannot clear Check 2 on trees either.

## What this rules out

Candidate A was chosen precisely because it makes no global-gradient assumption.
It still fails on trees. That removes the most plausible explanation for β's
0/6 — "the linear form was the problem" — and points the failure at something
about tree predictors themselves rather than the shape of the difficulty
estimator.
