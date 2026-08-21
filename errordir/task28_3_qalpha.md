# Task 28, Item 3 — The q_α(z) Estimator

**Built after Items 1–2 were complete and reviewed, with its purpose narrowed
per Item 1: not a novel-method contribution (the space is occupied by RLCP and
localized quantile regression) but a single empirical test of whether a learned
allocation closes the coverage gap the fixed multiplier could not.**

Script: `task28_3_qalpha.py`

## The estimator

`q_α(z)` = the (1−α) conditional quantile of the nonconformity score given the
difficulty coordinate `z = β′x`:

1. Sort calibration points by `z`.
2. Over a centred sliding window of `K` neighbours in z-order, take the empirical
   0.90 quantile of the scores in that window.
3. Enforce monotone non-decreasing in `z` by isotonic regression (PAVA).
4. Interpolate linearly between knots; clamp outside the observed z-range (no
   extrapolation).

**Why isotonic + window rather than linear quantile regression in z:**
monotonicity is required by the task and isotonic enforces it exactly rather than
approximately; the fit is non-parametric in *shape*, which matters because the
whole diagnosis from Task 27 was that the fixed multiplier imposed a shape
(linear in rank) — imposing a different fixed shape would repeat the mistake;
and PAVA introduces no additional smoothing constant.

## The one free parameter, fixed in advance

    K = max(30, ceil(n_cal / 10))

Rationale, stated from fit-period reasoning **before** any test-period result:

- **Floor of 30.** Below ~30 points a 0.90 quantile is determined by the top 3
  observations and is pure noise. 30 is the smallest window resting on ≥3 order
  statistics with any stability — the same reasoning Task 22 used to prefer p99
  over max at n=291.
- **n_cal/10.** Gives 10 effective difficulty levels, matching the resolution the
  fixed multiplier's rank map effectively had, so the comparison isolates
  **shape** (learned vs imposed) rather than resolution.

**No sweep over K against test data is run anywhere in this task.** This is the
test-set-tuning trap Task 22 established and every task since has respected.

## Correctness check

On a synthetic problem where spread grows with `z`
(`s ~ |N(0, 1 + 2·max(z,0))|`):

- monotone in z: **True**
- learned shape: q(−2)=1.436, q(−1)=1.546, q(0)=1.546, q(+1)=5.154, q(+2)=6.857 —
  correctly flat where noise is constant, rising where it grows
- conditional coverage on 4,000 fresh points: **88.0%** against a 90% target

## No-leakage

β comes from the FIT split only; `q_α(z)` is estimated on the CALIBRATION split
only; TEST is never touched during estimation. Item 4 validates on a CAL
subsplit disjoint from the one `q_α` was fit on.
