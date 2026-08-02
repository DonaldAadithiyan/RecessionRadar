# Task 9A — Extended Recession Test Window

**Verdict: the six-month ambiguity does NOT resolve. It narrows, and one arm
now resolves, but the headline out-of-fold comparison still straddles 90%.**

The test window grew from 65 to 79 months (n at six months: **59 → 71**) using
real new FRED observations. That is a meaningful increase, but not enough to
separate the six-month result from nominal.

Script: `fix-reg/task9a_extend_window.py`.
Data: `task9a_extended_recession_window.csv`, `task9a_extension_data.csv`.

---

## What was pulled

FRED's `RECPROUSM156N` now runs through **May 2026**, and the 12 indicator
series were refetched alongside it. The original panel ended May 2025.

- **14 new months** of usable target data (June 2025 – July 2026 by forecast
  date), slightly more than the ~11 the task anticipated.
- New scored n per horizon: Current 77, 1M 76, 3M 74, **6M 71** (was 59).

**No retraining.** Stage 1 / Stage 2 were not refit; the saved stacking ensemble
scored the new months exactly as it scored the original 65. The only refitting
is the out-of-fold selector arm's rolling-origin procedure extended forward —
the same procedure that produced the published OOF numbers, not a new kind.

## The regime check: the new months are a calm period, not a third regime

| | New months (2025-06 → 2026-07) | Original window (2020-01 → 2025-05) |
|---|---|---|
| recession prob, mean | 0.443 | 3.488 |
| recession prob, max | **1.16** | **100.00** |
| 3-month rate, mean | 3.792 | 2.626 |
| unemployment, mean | 4.317 | 4.891 |
| 10-year rate, mean | 4.254 | 2.823 |

The rate environment is genuinely different — short rates ~1.2pp higher and long
rates ~1.4pp higher than the original window's average — so the extension does
test the model under a higher-rate regime than most of its training data. But
**the new months contain no rare event at all**: peak recession probability is
1.16% versus 100% in the original window.

This matters for interpreting the result. The 12 added months are all
easy-to-cover expansion months, so they raise coverage slightly and tighten the
interval, but they add no information about the hard case — behaviour during a
rare-event transition, which is what the six-month deficit was always about.
**A larger n of calm months cannot resolve a question about turbulent ones.**

## A necessary caveat: the panel had to be rebuilt, and that is not free

The engineered features could not be extended incrementally — STL decomposition
is a global smoother, so its trend/residual outputs shift when re-run over a
longer series, and the quarterly GDP series' monthly interpolation shifts with
it. The entire panel was therefore rebuilt end-to-end from the raw FRED series
using the original recipe (STL seasonal=13/period=12; rolling stats on
`.shift(1)`; ±3σ anomaly thresholds fit on the original training partition only).

Reconstruction fidelity on overlapping months:

- median feature correlation **0.9997**; 30 of 46 features at r > 0.99
- but **13 of 46 at r < 0.8** — concentrated in STL residual/anomaly features
  and the quarterly GDP derivatives
- target series reproduces at r = 0.9983

Because the surviving differences propagate through the model, predictions on
the *same* months differ (6M: mean |Δ| = 7.1pp, r = 0.69). So the extended
numbers are **internally consistent but not directly comparable** to the
published ones. To separate the two effects, the same strategies were also run
on the rebuilt panel restricted to the original 65-month window:

| Strategy (6M, OOF) | Published (orig panel, n=59) | Rebuilt, same window (n=65) | Rebuilt, extended (n=71) | Reconstruction effect | New-data effect |
|---|---|---|---|---|---|
| Pooled/trailing | 84.75 | 86.15 | 85.92 | **+1.40pp** | **−0.23pp** |
| Diversity-optimal | 96.61 | 95.38 | 95.77 | **−1.23pp** | **+0.39pp** |

Both effects are small (≤1.4pp), and they point in opposite directions, so the
qualitative picture is unchanged by either. This is the check that makes the
extended result usable.

## The six-month verdict

| Arm | Strategy | n | Coverage | Wilson 95% | Verdict |
|---|---|---|---|---|---|
| in-sample | pooled/trailing | 71 | 81.69 | [71.15, 88.98] | **FALLS SHORT — resolved** |
| in-sample | diversity-optimal | 71 | 85.92 | [75.98, 92.17] | straddles 90% |
| out-of-fold | pooled/trailing | 71 | 85.92 | [75.98, 92.17] | straddles 90% |
| **out-of-fold** | **diversity-optimal** | **71** | **95.77** | **[88.30, 98.55]** | **straddles 90%** |

**One arm resolves:** the in-sample trailing baseline's interval is now entirely
below 90%, confirming that the paper's originally-reported six-month deficit is
real *under in-sample scoring*. That is a genuine tightening of a claim that was
previously undetermined.

**The headline does not.** The out-of-fold diversity-optimal result reaches
95.77% but its lower bound is 88.30 — still 1.7pp short of clearing nominal. The
out-of-fold baseline likewise still straddles. The paper's "undetermined under
honest scoring" language stands, now at n=71 rather than n=59.

**How much more data would settle it?** At the observed 95.77% rate, the Wilson
lower bound clears 90% at roughly n ≈ 100 scored months — about another 29
months, i.e. late 2028 at the current reporting cadence. That is the honest
answer to "how long until this resolves", assuming the rate holds.

## Full extended-window comparison (out-of-fold)

| Strategy | Current (n=77) | 1M (n=76) | 3M (n=74) | 6M (n=71) |
|---|---|---|---|---|
| Pooled/trailing ACI | 96.10 | 92.11 | 91.89 | 85.92 |
| Mondrian | 92.21 | 92.11 | 90.54 | 83.10 |
| PID-conformal | 96.10 | 92.11 | 91.89 | 87.32 |
| EVT-tail | 96.10 | 92.11 | 91.89 | 87.32 |
| DtACI | 96.10 | 92.11 | 91.89 | 87.32 |
| AcMCP | 97.40 | 89.47 | 87.84 | 73.24 |
| Bellman CI (untuned) | 81.82 | 82.89 | 81.08 | 57.75 |
| **Diversity-optimal (ours)** | **97.40** | **94.74** | **95.95** | **95.77** |

The strategy ordering is unchanged from the published window: diversity-optimal
highest at every horizon, AcMCP and Bellman CI undercovering, the standard
toolbox clustered near the baseline. **The extension confirms the existing
findings rather than revising any of them.**

## Honest caveats

- **The added months are all calm.** They increase n but add no rare-event
  transitions, which is precisely the regime the six-month deficit concerns.
  Treat this as a larger sample of the easy case.
- **Reconstruction is imperfect.** 13 of 46 features reconstruct at r < 0.8 and
  6M predictions on shared months correlate at only 0.69. The decomposition
  above bounds the impact at ≈1.4pp, but the extended numbers should be cited
  as "recomputed on a rebuilt panel", not as a drop-in update to Table 8.
- **Diversity-optimal's width grows too** (131.6 at 6M vs 51.5 for the
  baseline). The coverage-for-width tradeoff reported throughout the paper is
  unchanged, and at 2.6× the baseline width it remains substantial.
- **Nothing here is a retrain of the forecasting models**, so the paper's
  point-prediction section is untouched.
