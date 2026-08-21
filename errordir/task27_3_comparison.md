# Task 27, Item 3 — Head-to-Head: Does Tail-Targeting Close the Coverage Gap?

## Pre-registered criterion: **NOT MET, on either domain.**

The criterion was: β_tail narrows the coverage gap against diversity-optimal
**without** giving back the Winkler advantage over DtACI, on at least one domain.

| domain | β | **coverage gap vs divopt** | **Winkler advantage over DtACI** | criterion |
|---|---|---|---|---|
| climate | β_mean | +3.29 pp | +1.051 | — |
| climate | β_tail | **+3.70 pp** (wider) | **+0.505** (halved) | **fails both halves** |
| energy | β_mean | +3.25 pp | +0.927 | — |
| energy | β_tail | **+3.25 pp** (identical) | +1.017 (slightly better) | **fails: gap unchanged** |

On **climate** β_tail is worse on both axes — and in any case its β failed the
validation gate (Item 2), so that row cannot count.

On **energy**, where β_tail does validate, the coverage gap is **numerically
identical to three significant figures** (3.25 pp). Tail-targeting moved the
Winkler score slightly (+0.090) but did nothing whatsoever to the coverage gap
this task existed to close.

Script: `task27_3_comparison.py` · Data: `task27_3_comparison.csv`,
`task27_3_significance.csv`

## Full tables (CQR now included)

### Climate (n=243)

| method | coverage | width | ratio | Winkler mean | Winkler median |
|---|---|---|---|---|---|
| **errordir_mean** | 91.36 | 9.441 | 1.08 | **11.220** | 9.528 |
| errordir_tail | 90.95 | 9.773 | 1.11 | 11.766 | 9.817 |
| mondrian | 87.24 | 9.441 | 1.08 | 12.145 | **8.018** |
| evt_tail | 90.12 | 10.009 | 1.14 | 12.268 | 10.052 |
| dtaci | 90.12 | 9.924 | 1.13 | 12.271 | 10.170 |
| pooled_trailing | 90.95 | 10.349 | 1.18 | 12.306 | 10.394 |
| pid_conformal | 89.71 | 9.812 | 1.12 | 12.426 | 9.932 |
| diversity_optimal | **94.65** | 11.704 | 1.33 | 12.822 | 11.667 |
| **cqr** | 87.24 | 11.773 | 1.34 | 13.351 | 11.875 |
| bellman_ci | 76.54 | **7.552** | 0.86 | 13.846 | 8.046 |
| acmcp | 90.95 | 19.534 | 2.23 | 22.983 | 20.002 |

### Energy (n=400)

| method | coverage | width | ratio | Winkler mean | Winkler median |
|---|---|---|---|---|---|
| **errordir_tail** | 86.25 | 7.628 | 0.63 | **28.685** | 7.557 |
| errordir_mean | 86.25 | 7.395 | 0.61 | 28.775 | **6.974** |
| diversity_optimal | **89.50** | 10.075 | 0.83 | 29.314 | 10.501 |
| dtaci | 82.25 | 5.866 | 0.48 | 29.702 | 5.914 |
| **cqr** | **56.00** | 5.363 | 0.44 | 29.810 | 7.037 |
| mondrian | 73.00 | 4.122 | 0.34 | 29.960 | 3.505 |
| evt_tail | 84.50 | 7.112 | 0.58 | 30.232 | 7.713 |
| pooled_trailing | 83.00 | 6.791 | 0.56 | 30.616 | 7.222 |
| pid_conformal | 84.00 | 7.590 | 0.62 | 30.898 | 7.846 |
| bellman_ci | 68.00 | **3.385** | 0.28 | 35.440 | 3.997 |
| acmcp | 93.75 | 72.160 | 5.92 | 82.686 | 75.857 |

No arm is vacuous (all ratios ≤ 5.92, far below the ~100 threshold).

## Anti-gaming checks on the β_tail arm

Both domains are temporal, so the block sign-flip null applies (Task 26 verified
this distinction matters). BH correction is across all **m=36** cells.

Energy/β_tail, the only validated tail arm:

| baseline | mean diff | **win rate** | BH q |
|---|---|---|---|
| acmcp | −54.000 | 91.2% | 0.0000 |
| pid_conformal | −2.213 | 66.0% | 0.0000 |
| pooled_trailing | −1.930 | 48.8% | 0.0000 |
| evt_tail | −1.547 | 49.0% | 0.0000 |
| bellman_ci | −6.755 | 27.3% | 0.0000 |
| dtaci | −1.016 | **22.2%** | 0.0573 |
| mondrian | −1.274 | 21.0% | 0.1614 |
| **cqr** | −1.124 | 43.0% | 0.3795 |
| diversity_optimal | −0.629 | 82.5% | 0.3442 |

The pattern is essentially identical to β_mean's: significant wins over the
weaker baselines, sub-50% win rates against dtaci, mondrian and bellman_ci where
the mean advantage is carried by the tail rather than the typical hour.

**β_tail does not beat diversity_optimal significantly** (q=0.344) despite an
82.5% win rate — the selector's losses are concentrated in a few large ones.

## Why tail-targeting did not help — the likely reason

The multiplier is a **monotone rank map** of the projection. Coverage is
therefore driven by the *ordering* of test points by projected difficulty, not by
the projection's units. β_mean and β_tail produce highly similar orderings
because features predicting large mean error largely coincide with features
predicting tail membership. Changing the fitting target changes the direction's
scale and slightly its tilt, but not enough of the ordering for the rank-based
multiplier to allocate width differently.

That points at the actual binding constraint: the **±25% multiplier band**, not
the fitting target. Closing a 3.25pp coverage gap would require widening
high-difficulty intervals more than the band permits — and Task 22 already tested
raising that ceiling, which produced vacuous intervals and gamed the Winkler
mean. The frontier position looks structural rather than a fixable target-choice.
