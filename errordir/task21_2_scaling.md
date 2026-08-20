# Task 21, Item 2 — Width-Scaling Design

**All parameters below were fixed and written to `task21_2_scaling.py` BEFORE
any Item 3 result existed.**

## The function

```
s_t = β · x_t                     raw difficulty projection
u_t = F_CAL(s_t)                  mapped to [0,1] by the CAL split's own CDF
m_t = 0.75 + 0.50 · u_t           multiplier, linear in RANK
width_t = m_t · 2 · Q_C(q_t)      the ACI interval, scaled
```

## Why quantile-mapped rather than linear in the projection

A linear map is not scale-free — its effect depends on the arbitrary units and
spread of β·x, which differ by horizon, so one parameter set could not carry the
same meaning across horizons. The rank map is invariant to any monotone
transform of the projection and robust to outliers, which a linear map is not.

## Why F_CAL and not F_TEST

The CDF is estimated on the **CAL split only** and frozen before any test point
is seen. Using the test projections' own CDF would leak the test distribution
into interval construction.

## LO = 0.75, HI = 1.25 — and the constraint this deliberately imposes

A symmetric ±25% band around the unscaled interval. Symmetric about 1.0 so that
**under a uniform rank distribution the mean multiplier is exactly 1.0**.

This is the design's most important property: it makes it *impossible* for the
method to win on width by uniformly shrinking every interval. Any width
reduction must come from correlation between the multiplier and the realized
error — i.e. from genuinely differential allocation. It makes the test harder to
pass and the result harder to fake.

The floor of 0.75 (not 0) also bounds the damage a wrong β can do: a
confidently-wrong low projection produces an interval 25% narrower, not an
arbitrarily narrow one.

## What actually happened to that constraint

**The pinning assumption did not hold on the test period.** Test projections
drifted far above the calibration distribution: mean rank **0.812** rather than
~0.5, with 15.4% of test months projecting above *every* CAL month. The realized
mean multiplier was therefore **1.156**, not 1.0.

This is reported as a finding in Item 3 rather than corrected away, because
re-centring the multiplier on the test distribution would require seeing the
test projections — precisely the leak this design forbids. Any deployment would
face the same problem.
