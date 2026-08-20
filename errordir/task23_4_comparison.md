# Task 23, Item 4 — Comparison with Full Anti-Gaming Discipline

## Outcome: **no validated improvement — but for the first time in this line of work, also no sign of Task 22's trap.** Every directional indicator is favourable and nothing is statistically supportable.

6M only (the sole horizon validated in Item 2).

Script: `task23_4_comparison.py` · Data: `task23_4_comparison.csv`,
`task23_4_significance.csv`

## Results

| Method | coverage | mean width | **width/spread** | **vacuous?** | Winkler mean | **Winkler median** | mean mult |
|---|---|---|---|---|---|---|---|
| diversity_optimal | 84.75 | 53.279 | 89.4 | No | 122.774 | 57.035 | — |
| dtaci | 89.83 | 53.289 | 89.4 | No | 116.712 | 57.185 | — |
| **errordir_disag** | 88.14 | **51.247** | **86.0** | **No** | **116.216** | **56.797** | 1.130 |

## Significance (permutation only — pairs are a time series, not exchangeable)

| Baseline | mean Winkler diff | **win rate** | perm (shift) | perm (sign-flip) | **BH q** |
|---|---|---|---|---|---|
| diversity_optimal | −6.558 | **42.4%** | 0.113 | 0.201 | 0.401 |
| dtaci | −0.497 | **45.8%** | 0.330 | 0.468 | 0.468 |

**Nothing is significant.** Best BH q = 0.401. The mean Winkler edge over DtACI
(−0.497) is a rounding-level difference on a score of ~116.

Wilcoxon was deliberately **not** run. Task 22 showed it reporting p=0.0022 on
these data while both permutation nulls said 0.34–0.82, because it assumes
exchangeable pairs on an autocorrelated series with large outliers. Reporting it
here would manufacture significance the data does not support.

## The anti-gaming checks, as first-class results

Task 22's trap was: BH-significant mean gain + low win rate + width ratio past
~100. Checking each:

| Check | Task 22's failing arm | **Task 23** |
|---|---|---|
| width/spread ratio | **134.6 (vacuous)** | **86.0 — below threshold, and below both baselines** |
| Winkler median vs DtACI | **+33.2 worse** | **−0.39 better** |
| mean multiplier | 2.355 (inflating) | **1.130** |
| win rate vs DtACI | 28.8% | **45.8%** |
| BH-significant? | Yes (q=0.0033) | No (q=0.468) |

**This is not Task 22's pattern.** Task 22 bought a significant mean by
inflating width 50% past the vacuity threshold while making the typical month
worse. Task 23 produces the *narrowest* intervals of any arm tested (51.247, vs
53.28 for both baselines), improves the Winkler median rather than wrecking it,
and keeps the multiplier near the design's 1.0 target. The direction of every
indicator is right.

What it does not have is statistical support. Win rates of 42.4% and 45.8% mean
it still loses in the majority of months, and both permutation tests are
comfortably non-significant.

## Honest reading

This is the closest any arm across Tasks 21–23 has come to a genuine
frontier improvement: narrower width at comparable coverage, better on both
Winkler mean *and* median, no vacuity, no width inflation. If the effect were
real, this is what it would look like.

At n=59 with a 45.8% win rate against DtACI, this testbed cannot establish it.
The correct statement is **not validated**, not "nearly significant" — and per
the guardrail it is reported as such rather than as a success with caveats.
