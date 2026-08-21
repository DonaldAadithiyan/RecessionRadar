# Task 26, Item 3 — Comparison Under Both Split Conditions

## The Mondrian test, reported separately (the thing this task exists to measure)

| condition | errordir Winkler | Mondrian Winkler | **diff** | win rate | BH q | verdict |
|---|---|---|---|---|---|---|
| **temporal** | 113.739 | 114.426 | **−0.687** | 27.8% | 0.547 | errordir ahead, **not significant** |
| **random** | 95.731 | 99.571 | **−3.840** | 40.2% | 0.239 | errordir ahead, **not significant** |

**errordir beats Mondrian in BOTH conditions, and the margin is 5.6× LARGER
under the random split than the temporal one** — the opposite direction from the
pre-registered prediction. Neither margin is significant after BH correction.

Script: `task26_3_comparison.py` · Data: `task26_3_comparison.csv`,
`task26_3_significance.csv`, `task26_3_validation.csv`

## Validation gate — both conditions pass

| condition | n_fit | n_cal | n_test | fit r | held-out CAL r | angle PC1 | perturb pct | validated |
|---|---|---|---|---|---|---|---|---|
| temporal | 1500 | 1000 | 600 | 0.447 | **0.431** | 80.4° | 0.970 | **Yes** |
| random | 1500 | 1000 | 600 | 0.455 | **0.256** | 81.5° | 0.985 | **Yes** |

Both validate, so both proceed — no condition is excluded, and the comparison is
symmetric.

Worth noting: β generalizes *better* under the temporal split (held-out r 0.431
vs 0.256) despite performing *worse* against Mondrian there. Signal quality and
competitive standing move in opposite directions, which is itself evidence
against a simple "temporal split favours errordir" story.

## Full tables

### Temporal condition (n=600, target spread ≈ 305)

| method | coverage | width | ratio | Winkler mean | Winkler median |
|---|---|---|---|---|---|
| dtaci | 88.33 | 55.689 | 0.18 | **107.469** | 55.847 |
| **errordir** | 89.33 | 60.829 | 0.20 | 113.739 | 61.668 |
| mondrian | 86.17 | 55.933 | 0.18 | 114.426 | 50.055 |
| pid_conformal | 89.00 | 69.882 | 0.23 | 118.039 | 71.112 |
| pooled_trailing | 88.33 | 60.575 | 0.20 | 118.132 | 57.078 |
| evt_tail | 88.33 | 60.227 | 0.20 | 118.383 | 57.476 |
| diversity_optimal | 92.67 | 82.710 | 0.27 | 123.560 | 82.062 |
| bellman_ci | 78.33 | 41.584 | 0.14 | 124.376 | 47.363 |
| acmcp | 82.67 | 177.898 | 0.58 | 264.746 | 249.618 |

errordir places **2nd of 9**, behind DtACI (+6.270, q=0.976 — DtACI is clearly
better here). Significant wins over pooled_trailing (q=0.000), evt_tail
(q=0.000), acmcp (q=0.000), diversity_optimal (q=0.013), bellman_ci (q=0.012).

### Random condition (n=600)

| method | coverage | width | ratio | Winkler mean | Winkler median |
|---|---|---|---|---|---|
| pid_conformal | 89.17 | 50.031 | 0.22 | **95.200** | 47.624 |
| pooled_trailing | 92.00 | 55.036 | 0.24 | 95.647 | 52.005 |
| **errordir** | 93.50 | 61.792 | 0.27 | 95.731 | 64.507 |
| dtaci | 91.33 | 54.994 | 0.24 | 95.750 | 52.127 |
| bellman_ci | 86.17 | 45.500 | 0.20 | 97.484 | 53.515 |
| evt_tail | 93.33 | 61.736 | 0.27 | 97.566 | 60.345 |
| mondrian | 93.50 | 66.193 | 0.29 | 99.571 | 52.031 |
| diversity_optimal | 97.33 | 96.972 | 0.42 | 118.429 | 92.432 |
| acmcp | 88.83 | 103.929 | 0.45 | 148.285 | 118.762 |

errordir places **3rd of 9**, in a near-tie cluster (95.200 / 95.647 / 95.731 /
95.750 — a 0.55 spread across four methods). Only diversity_optimal (q=0.000)
and acmcp (q=0.000) are significantly beaten.

## Permutation validity — verified, not assumed

The guardrail required checking rather than blindly applying the same test. The
lag-1 autocorrelation of the paired Winkler differences:

| condition | range across baselines |
|---|---|
| temporal | **0.187 – 0.500** |
| random | **0.027 – 0.150** |

The temporal condition genuinely carries serial dependence and needs the
block sign-flip null. The random condition has little, so a plain paired
permutation would also be valid there — the block sign-flip null is used for
both anyway, since it remains valid under independence and keeps the two
conditions directly comparable. BH correction is applied across all **m=16**
cells from both conditions combined.

No arm in either condition is vacuous (all width ratios ≤ 0.58).
