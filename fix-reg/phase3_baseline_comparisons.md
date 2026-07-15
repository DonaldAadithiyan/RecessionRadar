# Phase 3 — Benchmark against existing conformal methods

## Phase 3a — Mondrian / class-conditional ACI (required)

**Verdict: Mondrian ACI does NOT solve the problem — it makes coverage *worse*
at every horizon. The well-known imbalanced-conformal fix backfires here, for a
concrete and reportable reason. This strengthens (does not threaten) the paper's
contribution.**

**Method.** Two regimes (expansion `rec_prob<50`, rare-event `≥50`). Maintain a
separate ACI state per regime; each test point uses the calibration-score
quantile of **its own regime**. Calibration = trailing N=254 (the paper's 40%
setting: 238 expansion + 16 rare-event scores). Test-regime taxonomy uses the
test month's true smoothed recession probability (an *oracle* grouping — the most
favorable case for Mondrian). Same test set/horizons/γ, no retraining.

| Horizon | Pooled cov | Pooled width | Mondrian cov | Mondrian width |
|---|---|---|---|---|
| Current | **89.23** | 7.45 | 81.54 | 3.65 |
| 1M | 85.94 | 7.35 | 85.94 | 5.84 |
| 3M | 85.48 | 14.95 | 85.48 | 6.85 |
| 6M | **67.80** | 10.64 | **61.02** | 8.23 |

**Why it backfires (the citable mechanism).** The test window is **63 expansion
vs. only 2 rare-event months**. Mondrian therefore applies the *narrow* expansion
quantile to 63/65 test points, so it stops "borrowing" the wide rare-event scores
that pooled ACI uses to widen every interval — Mondrian widths collapse (3M:
14.95→6.85) and coverage drops (Current −7.7pp, 6M −6.8pp). 1M/3M coverage is
unchanged only because those horizons were already at ceiling for a different
reason (see Phase 1). This is the classic small-conditioning-class failure of
Mondrian methods under extreme imbalance (50 rare / 581 total), and it holds
*even with oracle regime labels*.

**What this means for the paper's framing.** The contribution should be
positioned **against** Mondrian, not merely alongside it: the paper's finding is
that pooled-but-*diverse* calibration is what delivers coverage, and that
partitioning by regime (the obvious textbook fix) is counterproductive when the
protected regime is rare in the test period. A reviewer who asks "why not just
use Mondrian?" now has a direct numerical answer.

Data: `fix-reg/phase3a_mondrian.csv`.

## Phase 3b / 3c — Extra conformal baselines (completed)

**Verdict: neither a distribution-shift-robust ACI (PID-conformal) nor an
EVT-tail conformal fixes the coverage deficit — confirming the failure is a
calibration-*diversity* problem, not a weak-base-algorithm problem.** Calibration
= trailing N=254; same test set/horizons; no retraining.

| Horizon | Pooled cov (w) | **PID-conformal** cov (w) | **EVT-tail** cov (w) |
|---|---|---|---|
| Current | 89.23 (7.45) | 89.23 (8.79) | 87.69 (6.18) |
| 1M | 85.94 (7.35) | 87.50 (7.62) | 85.94 (6.14) |
| 3M | 85.48 (14.95) | 87.10 (17.42) | 85.48 (12.50) |
| 6M | 67.80 (10.64) | **71.19** (16.84) | 66.10 (9.46) |

- **3c PID-conformal** (Angelopoulos et al. 2024; ACI proportional α-update + an
  integral + derivative term). Nudges coverage up a little (6M +3.4pp to 71.2%)
  by widening intervals, but **6M stays ~19pp below nominal**. So the coverage
  deficit is *not* an artifact of vanilla ACI being weak — a purpose-built
  shift-robust controller barely helps. The bottleneck is the calibration score
  distribution, exactly the paper's claim.
- **3b EVT-tail conformal** (Generalized-Pareto tail fit to calibration-score
  exceedances over p80, quantile read from the fitted tail — an accessible
  stand-in for extreme-value CQR, which would require fitting quantile-regression
  models; flagged as such, not full EV-CQR). It slightly *hurts* 6M (66.1%),
  because extrapolating a tail from expansion-dominated calibration scores still
  underestimates the true spread. **You cannot model your way to a tail your
  calibration set never observed** — reinforcing that diversity must be present
  in the data, not synthesized post hoc.

**Combined Phase 3 takeaway.** Against three alternative conformal fixes —
Mondrian (worse), PID (marginal), EVT-tail (worse) — the diversity-optimal
selection from Phase 2 is the only approach that materially moves long-horizon
coverage. The paper's contribution is positioned *against* the standard toolbox,
not merely beside it. Data: `fix-reg/phase3bc_extra_baselines.csv`.
