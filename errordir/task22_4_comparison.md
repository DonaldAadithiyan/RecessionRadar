# Task 22, Item 4 — Re-run Comparison with Both Fixes Applied

## Outcome: **3 (no defensible improvement).** The mean-Winkler gain over DtACI is real but is bought entirely by a 50.5% width increase, and the method loses in 71.2% of individual months.

The surface reading looks like a win: `task22_both` beats DtACI on mean Winkler
(−2.570, Wilcoxon p=0.0022, BH q=0.0033). It is not a win. The win rate is
**28.8%**, mean width rises from 53.3 to **80.2 (+50.5%)**, and the Winkler
**median gets worse by +33.2**. This is the same mean-vs-win-rate confound Task 21
caught at 40.7%, in a more extreme form — and this time the mean is
statistically significant, which makes it more dangerous, not less.

Script: `task22_4_comparison.py` · Data: `task22_4_comparison.csv`,
`task22_4_significance.csv`

## Results (6M only)

| Method | coverage | mean width | Winkler mean | Winkler median | mean mult |
|---|---|---|---|---|---|
| diversity_optimal | 84.75 | 53.279 | 122.774 | 57.035 | — |
| dtaci | 89.83 | 53.289 | 116.712 | **57.185** | — |
| task21_original | 88.14 | 52.498 | 116.923 | 58.261 | 1.156 |
| fix1_only (rolling) | 88.14 | **51.854** | 117.018 | 57.494 | 1.143 |
| fix2_only (ceiling) | 96.61 | 81.904 | 114.970 | 92.218 | 2.407 |
| **task22_both** | **96.61** | **80.220** | **114.142** | **90.420** | 2.355 |

## Significance

| Method | Baseline | mean Winkler diff | **win rate** | Wilcoxon p | BH q | perm (shift) | perm (sign-flip) |
|---|---|---|---|---|---|---|---|
| task22_both | diversity_optimal | −8.632 | **32.2%** | 0.0018 | 0.0033 | 0.381 | 0.335 |
| task22_both | dtaci | −2.570 | **28.8%** | 0.0022 | 0.0033 | 0.821 | 0.342 |
| fix1_only | diversity_optimal | −5.756 | 40.7% | 0.7171 | 0.7171 | 0.059 | 0.227 |
| fix1_only | dtaci | +0.306 | 40.7% | 0.6398 | 0.7171 | 0.861 | 0.570 |
| fix2_only | diversity_optimal | −7.804 | 32.2% | 0.0007 | 0.0030 | 0.189 | 0.326 |
| fix2_only | dtaci | −1.742 | 28.8% | 0.0010 | 0.0030 | 0.166 | 0.442 |

**Note both permutation nulls are non-significant** (0.821 and 0.342 against
DtACI) even where Wilcoxon and BH say p<0.005. The Wilcoxon test assumes
exchangeable paired differences; these are a time series with a few enormous
2020 outliers. The permutation tests, which respect that structure, do not
confirm the result — exactly the discrepancy Task 20's discipline exists to
surface.

## Why the mean improves while the method gets worse

The Winkler score penalises a miss by `(2/α)·distance` = 20× the shortfall at
α=0.10. The four Jan–Apr 2020 months had errors of 24–47 against a half-width of
~26.6, so each generated a penalty in the hundreds. Raising the ceiling to 2.79
widens those specific intervals enough to avoid the penalties.

That rescues the **mean**. It does nothing for the other 55 months, which now
carry intervals 50% wider for no benefit — visible directly in the Winkler
**median rising from 57.2 to 90.4**. Mean improves, median worsens by more:
the definition of a result carried by a handful of points.

## The vacuity check

Using this project's own convention (width relative to the 6M target spread of
0.596):

| Method | width | ratio | |
|---|---|---|---|
| dtaci | 53.289 | 89.4 | |
| task21_original | 52.498 | 88.1 | |
| **task22_both** | **80.220** | **134.6** | **exceeds the vacuity threshold** |

The paper flags intervals as vacuous past a width/spread ratio of ~100 (Task 7
uses `width > 100` on a 0–100 scale). `task22_both` sits at 134.6 — it has bought
96.61% coverage with intervals wider than the plausible range of the target.
This is precisely the coverage-vacuity trap the project already documented.

## What each fix contributed

- **Item 1 (rolling recentering) alone**: genuinely helps on width (51.854, the
  narrowest of any arm) and cuts saturation from 15.4% to 3.1%. But it changes
  nothing statistically — win rate stays at 40.7%, all p ≥ 0.64. It is a correct
  fix to a real bug that turns out not to have been what limited the method.
- **Item 2 (raised ceiling) alone**: produces essentially the entire effect seen
  in `task22_both`. Both the apparent Winkler gain and the width blowout are
  attributable to it.

So the honest decomposition is: **the confound fix was correct and inert; the
ceiling fix traded a large width penalty for avoiding four catastrophic
penalties.** Neither produced the outcome-2 result this task was scoped to hope
for.
