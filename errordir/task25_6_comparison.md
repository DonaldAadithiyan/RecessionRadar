# Task 25, Item 6 — Full Comparison, Validated Fits Only

## Outcome, stated per Task 24's own precision standard: **energy/ridge wins the combined Winkler tradeoff (1st of 9); insurance/ridge does not (2nd of 9, behind Mondrian). NEITHER beats the best baseline on coverage and width simultaneously.**

Script: `task25_456.py` · Data: `task25_6_comparison.csv`,
`task25_6_significance.csv`, `task25_4_baselines.csv`

Validated fits only: insurance/ridge, energy/ridge (n=400 test points each).

## Energy / ridge — errordir ranks 1st of 9

| method | coverage | width | ratio | **Winkler mean** | Winkler median |
|---|---|---|---|---|---|
| **errordir** | 86.25 | 7.395 | 0.61 | **28.775** | 6.974 |
| diversity_optimal | 89.50 | 10.075 | 0.83 | 29.314 | 10.501 |
| dtaci | 82.25 | 5.866 | 0.48 | 29.702 | 5.914 |
| mondrian | 73.00 | 4.122 | 0.34 | 29.960 | 3.505 |
| evt_tail | 84.50 | 7.112 | 0.58 | 30.232 | 7.713 |
| pooled_trailing | 83.00 | 6.791 | 0.56 | 30.616 | 7.222 |
| pid_conformal | 84.00 | 7.590 | 0.62 | 30.898 | 7.846 |
| bellman_ci | 68.00 | 3.385 | 0.28 | 35.440 | 3.997 |
| acmcp | **93.75** | 72.160 | 5.92 | 82.686 | 75.857 |

| baseline | mean diff | **win rate** | perm p | **BH q** |
|---|---|---|---|---|
| acmcp | −53.911 | 91.5% | 0.0000 | 0.0000 |
| pid_conformal | −2.123 | 70.8% | 0.0000 | 0.0000 |
| evt_tail | −1.457 | 51.5% | 0.0000 | 0.0000 |
| pooled_trailing | −1.841 | 52.2% | 0.0005 | 0.0011 |
| bellman_ci | −6.665 | 27.8% | 0.0005 | 0.0011 |
| dtaci | −0.926 | **25.2%** | 0.0645 | 0.0867 |
| mondrian | −1.185 | **21.0%** | 0.1420 | 0.1623 |
| diversity_optimal | −0.539 | 82.0% | 0.3490 | 0.3723 |

**Beats simultaneously on coverage AND width:** pooled_trailing, pid_conformal,
evt_tail, mondrian, dtaci, bellman_ci — six of eight. Only acmcp has higher
coverage (93.75%), bought with a **vacuous-adjacent width ratio of 5.92**
(72.16 vs errordir's 7.40, nearly 10×). Only bellman_ci is narrower, at 68%
coverage — under-covering by 22 points.

**Where it falls short:** win rate is below 50% against dtaci (25.2%), mondrian
(21.0%) and bellman_ci (27.8%). Those three produce much narrower intervals and
win the majority of easy half-hours while losing badly on spikes — the mean is
carried by the tail. Against dtaci the mean advantage (−0.926) is **not
significant** (q=0.087), and against mondrian it is not either (q=0.162). The
significant wins are over pid_conformal, evt_tail, pooled_trailing, bellman_ci
and acmcp.

## Insurance / ridge — errordir ranks 2nd of 9, behind Mondrian

| method | coverage | width | ratio | **Winkler mean** | Winkler median |
|---|---|---|---|---|---|
| **mondrian** | 89.00 | 3.400 | 0.79 | **4.857** | 3.018 |
| errordir | 90.25 | 4.165 | 0.97 | 5.049 | 4.272 |
| dtaci | 89.25 | 4.226 | 0.98 | 5.184 | 4.311 |
| evt_tail | 89.00 | 4.173 | 0.97 | 5.216 | 4.222 |
| pid_conformal | 90.50 | 4.427 | 1.03 | 5.281 | 4.695 |
| pooled_trailing | 87.75 | 3.951 | 0.92 | 5.283 | 4.020 |
| diversity_optimal | **98.00** | 5.607 | 1.30 | 5.807 | 5.503 |
| bellman_ci | 73.25 | 2.663 | 0.62 | 6.648 | 3.365 |
| acmcp | 87.50 | 12.458 | 2.89 | 14.697 | 14.822 |

Significant wins over diversity_optimal (q=0.000, 90.8% win rate), acmcp
(q=0.000), bellman_ci (q=0.003), pid_conformal (q=0.050) and pooled_trailing
(q=0.038). **Loses to mondrian** (+0.193, win rate 16.5%, q=0.880 — Mondrian is
clearly better). Does not significantly beat dtaci (q=0.123) or evt_tail
(q=0.087).

Note insurance repeats Task 24's healthcare pattern exactly: **Mondrian wins,
errordir places 2nd.** Mondrian has now beaten errordir on both of the two
non-temporal, randomly-split datasets tested (healthcare, insurance) and lost on
both temporal ones (climate, energy). With n=4 that is a pattern worth naming,
not a result — but it is a specific, testable hypothesis for a future task.

## The precise claim, per axis

| dataset | best coverage | best width | best Winkler | errordir wins? |
|---|---|---|---|---|
| energy | acmcp 93.75 (ratio 5.92) | bellman_ci 3.385 (68% cov) | **errordir 28.775** | **combined only** |
| insurance | diversity_optimal 98.00 | bellman_ci 2.663 (73% cov) | mondrian 4.857 | **no** |

**Neither dataset produces a "beats every baseline on every axis" result.** Energy
produces a "wins the combined tradeoff" result; insurance produces a "beats most
baselines but loses to the best one" result.

## Skepticism checks applied (per the Item 6 guardrail)

- **Permutation construction verified non-degenerate.** Task 24 found the
  circular-shift null is vacuous for a mean statistic. Only the block sign-flip
  null is used here, and `nullb.std() > 1e-12` is asserted at runtime.
- **Calibration split is uniform.** Every method — baselines and errordir alike —
  calibrates on the same CAL half, so no method sees data another was denied.
  This is the correction Task 24's Finding 3 surfaced, applied from the start.
- **Vacuity checked as a first-class result.** errordir's ratios are 0.61 and
  0.97, far below the ~100 threshold. acmcp at 5.92 (energy) is the only arm
  approaching a width problem, and it is flagged in the table.
- **Win rate reported alongside every mean**, and the sub-50% cases are stated
  in the text rather than buried.
