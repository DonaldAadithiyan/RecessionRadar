# Task 21, Item 4 — Where β Is Confidently Wrong

**Ran regardless of Item 3's outcome, per the guardrail. Result: the method's
distinctive risk — an artificially narrow interval that then misses — did NOT
materialize on this test period. All 12 intervals the method narrowed were
correctly targeted. But a separate, nameable failure mode did appear, clustered
100% in the COVID onset.**

Script: `task21_4_failure.py` · Data: `task21_4_failures.csv`

## The distinctive risk, tested directly

The method narrowed **12 of 59** months (multiplier < 1.0):

| | months narrowed | mean realized error |
|---|---|---|
| narrowed (mult < 1.0) | 12 | **1.35** |
| not narrowed | 47 | **13.33** |

Narrowing was **correctly targeted** — a 10× lower realized error in exactly the
months the method chose to narrow. No confidently-wrong narrow interval occurred.
This is the one clearly positive result in Task 21: where β was confident enough
to shrink an interval, it was right to.

Narrowed months fall in 2020 (8, mean multiplier 0.750) and 2021 (4, mean 0.801).

## The failure mode that did appear

Applying the pre-specified tercile definition (lowest-rank ∩ highest-error) flags
4 of 59 months (6.8%), and they cluster perfectly:

| Date | rank | multiplier | realized error | pred | actual |
|---|---|---|---|---|---|
| 2020-01 | 0.965 | 1.232 | 24.27 | 24.29 | 0.02 |
| 2020-02 | 0.913 | 1.207 | 23.69 | 23.69 | 0.00 |
| 2020-03 | 0.941 | 1.220 | 46.98 | 46.98 | 0.00 |
| 2020-04 | 0.980 | 1.240 | 14.14 | 14.18 | 0.04 |

**100% fall in Jan–Apr 2020** — the COVID onset. By year:

| year | n | confidently-wrong | mean error |
|---|---|---|---|
| 2020 | 12 | 4 (33.3%) | 9.98 |
| 2021 | 12 | 0 | 1.78 |
| 2022 | 12 | 0 | 30.69 |
| 2023 | 12 | 0 | 3.74 |
| 2024 | 11 | 0 | 8.05 |

**A correction to how these should be read.** These four months received
multipliers of **1.21–1.24** — the method made these intervals *wider*, not
narrower. The tercile definition labels them "low rank" only because covariate
drift pushed the bottom-tercile threshold up to 0.984; in absolute terms these
are high-difficulty projections. So they are **not** instances of the
confidently-wrong-narrow-interval risk. They are months where the underlying
ensemble was catastrophically wrong (predicting 24–47 against an actual of ~0)
and β correctly flagged them as difficult, but the ±25% band was far too small to
compensate.

That is the honest nameable failure mode: **β's difficulty ranking survives a
regime break, but the bounded multiplier cannot scale far enough to matter when
the base model fails badly.** A 24% width increase against a 47-point error is
irrelevant. This is a limitation of the scaling band, not of β.

Note 2022 has the highest mean error of any year (30.69) with zero flagged cases
— β did *not* anticipate that year's errors at all. So the diagnostic is not
uniformly reliable either.

## Comparison to the standard the paper holds others to

Mondrian's regime-imbalance failure and EVT-tail's threshold sensitivity are both
structural and nameable. This method's is too, and it has two parts:

1. **Bounded compensation.** The ±25% band is too narrow to matter when the base
   model fails at the scale seen in early 2020. Widening the band would help
   there but would also deepen the confidently-wrong-narrow risk elsewhere — a
   real tradeoff, not a free parameter.
2. **Rank-map drift.** The multiplier's meaning depends on the calibration
   projection distribution holding at test time. It did not (Item 3), and this
   silently converted a width-neutral design into a ~16%-widening one.
