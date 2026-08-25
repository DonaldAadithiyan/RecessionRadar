# Task 33, Item 3 — Locally-Varying Difficulty Function D(x)

## Result: **0/6 validated — and the ridge control fails too (0.921, 0.890), so this does NOT isolate a tree-specific limitation. The test is also compromised by a bandwidth that made D(x) effectively global rather than local.**

Script: `task33_3_local_difficulty.py` · Data: `task33_3_local_difficulty.csv`

## Construction

Nadaraya-Watson kernel regression of |error| on x in standardized FIT space,
with the gradient in **closed form** (not finite differences, which would add a
step-size parameter to justify):

    D(x)      = Σ K_h(x,x_i) e_i / Σ K_h(x,x_i)
    ∇D(x)     = (2/h²)·[ weighted-mean(e·offset) − D(x)·weighted-mean(offset) ]

Chosen over splines/NNs because it has a closed-form gradient, is local by
construction, and introduces exactly **one** free parameter. `h` = median
pairwise distance among FIT points, computed FIT-only — the same estimator-free
default Task 29 used for RLCP. No sweep over `h` was run.

**Check 2 was applied per-point**: each test point perturbed along **its own**
∇D(x), against **its own** random-direction null. That was the mechanistic
difference from Tasks 21–32, where one direction per fit was tested.

## Results

| domain | model | class | h | held-out r | angle PC1 | Check 1 | mean pct | pts ≥.95 | validated |
|---|---|---|---|---|---|---|---|---|---|
| insurance | xgboost | tree | 4.357 | 0.041 | 55.8° | Pass | 0.503 | 0.03 | **No** |
| insurance | lightgbm | tree | 4.357 | 0.044 | 57.1° | Pass | 0.497 | 0.03 | **No** |
| insurance | **ridge** | linear | 4.357 | 0.055 | 73.7° | Pass | **0.921** | 0.40 | **No** |
| energy | xgboost | tree | 4.210 | 0.363 | 28.5° | **Fail** | 0.671 | 0.05 | **No** |
| energy | lightgbm | tree | 4.210 | 0.370 | 26.9° | **Fail** | 0.693 | 0.10 | **No** |
| energy | **ridge** | linear | 4.210 | 0.342 | 22.2° | **Fail** | 0.890 | 0.35 | **No** |

## Two findings that limit what this result can be read as

**1. The ridge control fails — unlike Task 32.** In Task 32 the local-kNN
candidate validated on both ridge arms (0.965, 1.000), which is what let that
task attribute the tree failures to model class. Here ridge reaches only 0.921
and 0.890. **This mechanism is weak on linear models too**, so its tree failures
say nothing specific about trees. The control did exactly the job it exists for:
it prevented a mechanism-level failure being reported as a model-class finding.

**2. The bandwidth made D(x) effectively global.** In standardized space with
d = 12–21 features, the median pairwise distance is √(2d) ≈ 4.9–6.5. The fitted
`h` is **4.21–4.36** — the same order as the typical inter-point distance. A
Gaussian kernel at that width averages over most of the sample, so D(x) is close
to a global mean-error surface, and ∇D(x) varies little across points.

That is visible in the results: per-point percentiles cluster tightly (only
3–10% of tree points exceed 0.95), which is what a near-constant gradient
produces. **The locality hypothesis was therefore not really tested** — the
estimator was configured, by its own principled default, into a nearly global
regime.

**I am not re-running with a smaller h.** The median heuristic was fixed in
advance from a stated rationale, and choosing a bandwidth after seeing that the
first one failed — by checking which value produces a better test-period
percentile — is precisely the test-set tuning this project has refused since Task
22. The honest report is that this configuration failed and that the
configuration itself was likely wrong for the question.

**3. Check 1 fails on energy** (angles 22–28°, below the 30° bar) — for ridge as
well as trees. The mean gradient direction there sits close to PC1, meaning D(x)
is largely tracking the dominant variance direction rather than a distinct
difficulty direction. Another symptom of the over-wide bandwidth.

## What would make this a real test

A genuinely local D(x) needs `h` at a fraction of the median inter-point
distance — but selecting that fraction requires a principled, fit-period-only
criterion (e.g. leave-one-out likelihood on the FIT split, or Silverman's rule
adjusted for dimension), not a test-period sweep. That is a well-defined follow-up
with a stated selection rule fixed in advance. It is **not** a fourth mechanism —
it is the same mechanism, correctly configured.
