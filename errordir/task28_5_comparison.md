# Task 28, Item 5 — Full Frontier Comparison (Expanded Baseline Set)

## Outcome, at this project's precision standard: **on climate, `q_α(z)` wins the combined tradeoff and ties RLCP — it does NOT beat every baseline on both axes. On energy it fails outright (10th of 12, 68.50% coverage).**

RLCP was added to the baseline set per Item 1's literature finding, implemented
in the **same 1-D coordinate** errordir uses, so the comparison isolates "learned
monotone quantile function" vs "kernel reweighting" on identical geometry.

Script: `task28_45.py` · Data: `task28_5_comparison.csv`, `task28_5_significance.csv`

## Climate (n=243)

| method | coverage | width | ratio | **Winkler mean** | Winkler median |
|---|---|---|---|---|---|
| **errordir_qalpha** | 89.30 | **8.666** | 0.99 | **10.770** | 7.939 |
| **rlcp** | 91.77 | 9.124 | 1.04 | **10.799** | 9.036 |
| errordir_fixed | 91.36 | 9.441 | 1.08 | 11.220 | 9.528 |
| mondrian | 87.24 | 9.441 | 1.08 | 12.145 | **8.018** |
| evt_tail | 90.12 | 10.009 | 1.14 | 12.268 | 10.052 |
| dtaci | 90.12 | 9.924 | 1.13 | 12.271 | 10.170 |
| pooled_trailing | 90.95 | 10.349 | 1.18 | 12.306 | 10.394 |
| pid_conformal | 89.71 | 9.812 | 1.12 | 12.426 | 9.932 |
| diversity_optimal | **94.65** | 11.704 | 1.33 | 12.822 | 11.667 |
| cqr | 87.24 | 11.773 | 1.34 | 13.351 | 11.875 |
| bellman_ci | 76.54 | **7.552** | 0.86 | 13.846 | 8.046 |
| acmcp | 90.95 | 19.534 | 2.23 | 22.983 | 20.002 |

**`q_α(z)` improves on the fixed multiplier**: Winkler 11.220 → **10.770**, width
9.441 → **8.666** (−8.2%). It is the narrowest non-degenerate method in the table.

### Significance (BH across all m=40 cells)

| baseline | mean diff | win rate | **BH q** |
|---|---|---|---|
| cqr | −2.581 | 90.5% | 0.0000 |
| acmcp | −12.214 | 81.5% | 0.0000 |
| diversity_optimal | −2.052 | 79.8% | 0.0000 |
| bellman_ci | −3.076 | 65.4% | 0.0000 |
| evt_tail | −1.498 | 75.3% | 0.0020 |
| pid_conformal | −1.656 | 73.3% | 0.0027 |
| dtaci | −1.501 | 73.3% | 0.0027 |
| pooled_trailing | −1.536 | 78.2% | 0.0035 |
| mondrian | −1.376 | 65.0% | 0.0475 |
| **rlcp** | **−0.029** | 72.4% | **0.5961** |

**Nine of ten baselines are beaten significantly, all with win rates above 50%.**
The tenth is RLCP — a statistical tie (−0.029 on a Winkler of ~10.8, q=0.596).

### The per-axis verdict (climate)

| axis | best baseline | q_α(z) | |
|---|---|---|---|
| coverage | diversity_optimal 94.65 | 89.30 | **loses** |
| width | bellman_ci 7.552 (76.5% cov) | 8.666 | **loses** |
| Winkler | rlcp 10.799 | **10.770** | **wins, but a tie** |

**Not universal dominance.** And notably, the coverage gap against
diversity_optimal *widened* (5.35 pp vs the fixed multiplier's 3.29 pp) — the
mechanism this task was built to close the gap actually traded coverage for
width, moving *along* the frontier rather than beyond it.

## Energy (n=400) — the redesign fails

| method | coverage | width | ratio | Winkler mean |
|---|---|---|---|---|
| errordir_fixed | 86.25 | 7.395 | 0.61 | **28.775** |
| diversity_optimal | 89.50 | 10.075 | 0.83 | 29.314 |
| dtaci | 82.25 | 5.866 | 0.48 | 29.702 |
| cqr | 56.00 | 5.363 | 0.44 | 29.810 |
| rlcp | 72.75 | 3.899 | 0.32 | 29.951 |
| mondrian | 73.00 | 4.122 | 0.34 | 29.960 |
| evt_tail | 84.50 | 7.112 | 0.58 | 30.232 |
| pooled_trailing | 83.00 | 6.791 | 0.56 | 30.616 |
| pid_conformal | 84.00 | 7.590 | 0.62 | 30.898 |
| **errordir_qalpha** | **68.50** | 3.775 | 0.31 | **30.987** |
| bellman_ci | 68.00 | 3.385 | 0.28 | 35.440 |
| acmcp | 93.75 | 72.160 | 5.92 | 82.686 |

`q_α(z)` drops to **68.50% coverage** — 21.5 points below nominal — and falls to
10th of 12, *worse* than the fixed multiplier it replaced (28.775 → 30.987).

**Item 4 predicted this.** The held-out bin check flagged energy bin 1 at 81.88%
before this comparison ran. Energy's test period is a different price regime
(Task 27: test mean 7.39 vs FIT 6.40), so a quantile function estimated on
calibration data systematically under-provisions when the test distribution
shifts. The fixed multiplier is anchored to the *live ACI quantile*, which adapts
online; `q_α(z)` replaces that anchor with a frozen estimate and loses the
adaptivity. That is the mechanism, and it is the same failure Task 27 documented
for CQR on this domain.

No arm in either domain is vacuous (all ratios ≤ 5.92).
