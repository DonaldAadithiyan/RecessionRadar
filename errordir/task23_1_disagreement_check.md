# Task 23, Item 1 — Is Ensemble Disagreement a Legitimate Signal?

**All three checks PASS. Proceed to Item 2 — with one caveat on Check 3 that
matters for interpreting later items.**

Script: `task23_1_disagreement_check.py` · Data: `task23_1_disagreement.csv`,
`task23_1_leakcheck.txt`

## The quantity

Stage 2 runs three base models (CatBoost / LightGBM / RandomForest) whose
per-horizon predictions the ElasticNet meta-learner combines. Per horizon:

    disag_std     = std(CatBoost, LightGBM, RandomForest)
    disag_range   = max(...) − min(...)
    disag_maxpair = max pairwise |difference|

These are **already computed inside the ensemble** as meta-features
(`ensemble_stubs.FullChainStackingEnsemble._engineer_meta_features` builds
`np.std`, `np.min`, `np.max`, and pairwise `np.abs` differences). This task
extracts existing quantities rather than inventing a new one.

## Check 1 — available at prediction time: PASS

A base-model prediction is a function of `X` alone; it never touches `y`. So
disagreement at month *t* is computable at prediction time for exactly the same
reason the prediction itself is.

| test | result |
|---|---|
| unchanged when all **future** rows corrupted | True |
| unchanged when all **past** rows corrupted | True |
| computation touches any outcome `y` | **NO** |

The past-corruption test establishes something stronger than the usual leakage
check: disagreement at *t* is **row-local**, depending on `X[t]` alone. It cannot
leak across time in either direction.

*(The first run reported FAIL on the outcome-grep. That was a self-referential
false positive — the only matching line was the check's own source. Same class
of bug as Task 22 Item 1's leak-grep; corrected to exclude the check line.)*

## Check 2 — continuous, computed every month: PASS

| | |
|---|---|
| fit-pool months | 635 |
| months with all features finite | **635 (100%)** |
| distinct values of `disag_std_6M` | **635 (100%)** |
| fraction exactly zero | 0.0000 |

Fully continuous, defined every month, never conditioned on a rare label. This
is the property that let β avoid the scarcity trap in Task 21, and it holds here
for the same reason.

## Check 3 — not redundant: PASS, but read the margin

Criterion fixed in advance: redundant iff R² from all existing features > 0.95.

| horizon | max abs corr | nearest existing feature | **R² from all existing** |
|---|---|---|---|
| Current | 0.708 | unemployment_rate_diff3 | **0.870** |
| 1M | 0.681 | gdp_per_capita_diff3 | **0.887** |
| 3M | 0.636 | gdp_per_capita_diff3 | **0.839** |
| **6M** | 0.507 | OECD_CLI_index_diff1 | **0.675** |

Worst case 0.887, comfortably under the 0.95 bar, so the check passes as
specified. **But the margin is thinner than "PASS" suggests**: at 1M, ~89% of
disagreement's variance is already reconstructible from features β can see, so
the genuinely new information there is about 11%.

**6M is the standout** — R²=0.675, meaning roughly a third of its content is
independent of the existing feature set. That is the horizon Task 21 validated
and the one Task 22's 2022 diagnosis concerns, so the new information is
concentrated exactly where it is needed. It also sets a realistic expectation:
if adding this feature helps anywhere, 6M is where, and the effect is bounded by
how much of that 32.5% is actually predictive of error.

## Gate

check1 = PASS, check2 = PASS, check3 = PASS → **proceed to Item 2.**
