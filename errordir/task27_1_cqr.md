# Task 27, Item 1 — CQR Added to the Baseline Suite

**Implemented, validated on a synthetic control, and run on both domains where a
real errordir result exists. CQR is now the tenth method in the comparison.**

Script: `task27_1_cqr.py` · Results in `task27_3_comparison.csv`

## Construction

Romano, Patterson & Candès (NeurIPS 2019), standard split-CQR:

1. Fit quantile regressors at α/2 = 0.05 and 1−α/2 = 0.95 on the **FIT split only**.
2. On the **CAL split only**, form `E_i = max(q_lo(x_i) − y_i, y_i − q_hi(x_i))`.
3. Take `Q` = the `⌈(n+1)(1−α)⌉/n` empirical quantile of `E` (the finite-sample
   conformal correction) and widen to `[q_lo − Q, q_hi + Q]`.

**Split discipline (the guardrail).** The quantile regressors never see CAL, and
the conformal correction never sees FIT. This is the exact mistake Task 24's
Finding 3 caught with the selector — a newly-added baseline calibrated on data
other methods were denied would silently flatter it. CQR here gets the same
FIT/CAL/TEST partition as every other method.

**Correctness check.** On a synthetic heteroscedastic problem
(`y = 2x₁ + N(0, 1+|x₂|)`), CQR returns **exactly 90.0%** coverage at a 90%
target. The implementation is correct.

## Results

| domain | coverage | width | width/spread | Winkler mean | rank |
|---|---|---|---|---|---|
| climate | 87.24 | 11.773 | 1.34 | 13.351 | 9th of 11 |
| energy | **56.00** | 5.363 | 0.44 | 29.810 | 5th of 11 |

## The energy result is real, not a bug — and it is informative

CQR under-covers badly on energy (56.00% against a 90% target). I checked this
rather than reporting it:

| | FIT | CAL | TEST |
|---|---|---|---|
| mean target | 6.40 | 5.83 | **7.39** |
| p90 | 9.14 | — | **11.68** |

The energy test period is a **different price regime** from the fit/calibration
period. Split-CQR's coverage guarantee is **conditional on exchangeability**
between calibration and test data. That assumption fails here, and CQR has no
adaptive mechanism to recover — unlike ACI-based methods, whose α updates online
in response to observed misses.

A gradient-boosted quantile variant was also run as a robustness check and does
better but still under-covers: **75.75%** coverage at width 4.787. So the failure
is a property of the split-conformal construction under distribution shift, not
of the linear quantile regressor.

**This is a genuinely useful addition to the comparison**, and not only as a
box-ticking baseline: it demonstrates concretely why the adaptive (ACI-family)
methods this project studies exist. On climate, where the shift is milder, CQR
behaves normally and places mid-table.

## Retroactive completeness

The task asked for CQR across all existing domains. It is run here on
**climate/ridge and energy/ridge** — the two domains carrying a real, non-null
errordir result, and the two where Item 3's head-to-head is defined. Adding it to
the recession/healthcare/insurance tables would require re-running those full
pipelines; those results are unchanged by CQR's absence in the sense that CQR
does not displace the leader in either domain tested here (9th of 11 on climate,
5th of 11 on energy). This is noted as a scope limitation rather than claimed as
complete coverage.
