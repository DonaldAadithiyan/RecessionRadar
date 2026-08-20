# Task 21, Item 1 — Fitting and Validating the Error-Direction β

**Result: β validates at 6M ONLY (1 of 4 horizons). Current, 1M, and 3M fail the
perturbation check. Per the guardrail, Items 2–3 proceed for 6M alone.**

Script: `task21_1_beta.py` · Data: `task21_1_beta.csv`, `task21_1_perturbation.csv`

## The separation that preserves the guarantee (verified, not assumed)

The pool (pre-2020, 635 months) is normally used *entirely* as the calibration
pool. Since β must not see calibration data, it was split temporally three ways:

| Split | Range | n | Role |
|---|---|---|---|
| FIT | first 60% of pool | 291 | β fit here, and **only** here |
| CAL | last 40% of pool | 254 | calibration scores; β never sees these |
| TEST | 2020+ | 59–65 | β never sees these |

Disjointness is asserted at runtime, not assumed from the pipeline. β is fit as a
**continuous ridge regression of |error| on all 46 features using every FIT
month** — not a classifier on rare large-error labels. That is the design choice
the proposal rests on, and it worked as intended: 291 training examples instead
of the 2–3 a rare-label classifier would have had.

## Results

| Horizon | r (fit) | r (held-out CAL) | angle(β, PC1) | perturb pct | Check 1 | Check 2 | Validated |
|---|---|---|---|---|---|---|---|
| Current | 0.912 | **0.109** | 89.4° | 0.370 | PASS | FAIL | **No** |
| 1M | 0.907 | **0.151** | 88.5° | 0.595 | PASS | FAIL | **No** |
| 3M | 0.803 | 0.349 | 74.4° | 0.925 | PASS | FAIL | **No** |
| 6M | 0.670 | **0.367** | 57.9° | **0.975** | PASS | **PASS** | **Yes** |

**Check 1 (not-just-variance) passes everywhere.** β sits 57.9°–89.4° from PC1,
so it is not the generic top-variance direction. Criterion (>30°) was fixed
before running.

**Check 2 (perturbation) fails at three of four horizons.** β had to sit above
the 95th percentile of a 200-direction random null; it reached only the 37th
percentile at Current and 59.5th at 1M. Criterion fixed before running.

## The pattern in the failures is informative

Check 2's percentile rises monotonically with horizon — **0.370 → 0.595 → 0.925
→ 0.975** — and tracks the held-out CAL correlation (0.109 → 0.151 → 0.349 →
0.367) in lockstep. This is a signal-strength gradient, not random failure: error
magnitude is genuinely more predictable from inputs at longer horizons, and only
at 6M is it predictable enough to clear the bar.

The fit-vs-held-out gap is the other half of the story. At Current, β fits the
FIT split at r=0.912 but generalizes at r=0.109 — near-total overfitting. At 6M
the gap is far smaller (0.670 → 0.367). Reporting the fit correlation alone would
have made all four horizons look excellent; the held-out column is what
distinguishes them, which is exactly why the separation was built first.

## Gate decision

Per the guardrail — "if either check fails, stop here and report that; do not
proceed to Item 2 with an unvalidated β" — Current, 1M, and 3M are excluded from
Items 2–3. Only 6M proceeds. This is a genuine narrowing of the proposal's scope,
not a technicality: three quarters of the horizons have no validated difficulty
signal.
