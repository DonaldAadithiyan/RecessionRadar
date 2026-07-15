# Mechanistic Evidence for the Calibration Threshold Finding

## P0 — Nonconformity scores by regime

**Score definition (confirmed from `aci_composition_sweep.py:225`):** the
nonconformity score is the **un-normalized absolute residual**
`|actual − predicted|` (percentage-point units), computed on the
calibration/training pool from the saved ensemble. ACI sets interval half-width
to `np.quantile(scores, 1 − α_t)` with `α_target = 0.10`, so the operative
quantile is the **90th percentile** (α_t drifts adaptively around 0.10, but p90
is the correct anchor). Regime label = smoothed recession probability ≥ 50%
("rare-event") vs. < 50% ("expansion"), 50 rare vs. 581 expansion months.

**Per-horizon nonconformity scores by regime (full calibration pool, pp):**

| Horizon | Exp. mean | Exp. median | Exp. p90 | Rare mean | Rare median | Rare p90 | Rare/Exp mean |
|---|---|---|---|---|---|---|---|
| Current | 1.40 | 0.86 | 1.61 | 5.00 | 4.50 | 7.33 | **3.58×** |
| 1M | 1.64 | 0.83 | 2.90 | 5.17 | 3.26 | 11.74 | **3.16×** |
| 3M | 2.72 | 1.81 | 3.56 | 9.26 | 9.56 | 18.89 | **3.40×** |
| 6M | 2.44 | 1.34 | 3.14 | 4.24 | 2.19 | 10.16 | **1.73×** |

**Separation is real, not overstated.** Only **4.5–12%** of expansion scores
exceed the rare-event *median* (Current 5.0%, 1M 9.3%, 3M 4.5%, 6M 12.0%) — so
"rare-event scores are ~3× larger" is an honest claim for Current/1M/3M. **6M is
the weakest case**: ratio only 1.73× and the highest overlap (12%), i.e. its
rare-event scores are less cleanly separated — consistent with 6M never reaching
nominal coverage in the sweep.

**The 20% → 25% transition — the exact mechanistic step.** Growing the trailing
window from 20% to 25% admits 31 older months, **15 of them 2008-recession
rare-event months**. Effect on the p90 quantile that sets ACI width:

| Horizon | p90 @ 20% | p90 @ 25% | Δ | Max rare score newly entered |
|---|---|---|---|---|
| Current | 1.05 | 2.90 | **+1.85** | 15.83 |
| 1M | 0.94 | 1.52 | +0.58 | 10.71 |
| 3M | 1.86 | 8.43 | **+6.57** | 18.15 |
| 6M | 1.54 | 2.48 | +0.94 | 26.32 |

The p90 jump is largest exactly where coverage jumped in the sweep (3M, Current).
At 20% the window (2009-06 → 2019-12) is a pure expansion with essentially no
recession months, so the p90 is tiny (~1–2 pp) and intervals are far too narrow;
once the 2008 episode enters at 25%, the p90 leaps and intervals widen enough to
recover coverage. 6M's p90 barely moves (+0.94) despite a large single score
(26.3) — one extreme value doesn't move the 90th percentile much — which is why
6M coverage stays stuck.

**Files:** `nonconformity_by_regime_stats.csv`, `nonconformity_transition_20_25.csv`,
`nonconformity_scores_raw.csv`. No models retrained (saved ensemble only).

---

## P1 — Nonconformity distribution figure

**Built:** `figures/nonconformity_by_regime.pdf` + `.png` (two-column, 7.0×2.3in),
underlying data `fix-reg/nonconformity_scores_raw.csv`.

**Plot style chosen: boxplots** (not overlaid histograms) on a **log y-axis**,
one facet per horizon, expansion (blue) vs. rare-event (orange, hatched for
grayscale-safety), with each group's **90th-percentile drawn as a dashed line**
(the quantile ACI uses to set interval width). Boxplots won over histograms
because the two groups are very unequal in size (581 vs. 50) and span two orders
of magnitude — overlaid histograms were lopsided and unreadable at 3.3in;
boxplots show the median/IQR shift and the p90 gap cleanly. All 4 horizons fit
legibly as small multiples, so all 4 are shown. The figure visually confirms P0:
rare-event boxes and their p90 lines sit well above expansion across all
horizons, most starkly at Current and 3M, weakest at 6M.

---

## P2 — Controlled fixed-size ablation (N=254, vary rare-event count)

**Verdict: the sharp threshold does NOT survive a fixed-size test. Holding
N=254 constant, coverage rises only *gradually* with rare-event count (Current,
6M) or not at all (1M, 3M) — and the result is dominated by *which* expansion
months fill the window, not by rare-event presence. The dramatic jump in the
original 20→25% sweep was substantially a window-*size* effect (20% = 127 rows)
confounded with composition, not composition alone.**

Construction (feasible; 50 rare + 585 expansion months available): calibration
set = k most-recent rare-event months + (254−k) most-recent expansion months.
Same test set, same ACI (γ=0.005, α=0.10), saved ensemble only.

**Fixed-size ablation — coverage (%) [recent-expansion fill]:**

| Rare months | Comp. % | Current | 1M | 3M | 6M |
|---|---|---|---|---|---|
| 0 | 0.0 | 81.5 | 85.9 | 85.5 | 62.7 |
| 1 | 0.4 | 83.1 | 85.9 | 85.5 | 62.7 |
| 4 | 1.6 | 83.1 | 85.9 | 85.5 | 62.7 |
| 8 | 3.1 | 84.6 | 85.9 | 85.5 | 66.1 |
| 16 | 6.3 | 89.2 | 85.9 | 85.5 | 67.8 |

At fixed N=254, Current climbs gradually (81.5→89.2, no sharp step), 6M climbs
mildly (62.7→67.8), and **1M/3M are already at ~85–86% with ZERO recession
months** — composition adds nothing there.

**Robustness — the result depends on the expansion fill (reported honestly):**

| Fill | Cur @0 rare | Cur @16 | 6M @0 | 6M @16 |
|---|---|---|---|---|
| recent expansion | 81.5 | 89.2 | 62.7 | 67.8 |
| oldest expansion | 89.2 | 89.2 | 78.0 | 78.0 |
| random expansion (seed 42) | 87.7 | 89.2 | 71.2 | 71.2 |

With **oldest** or **random** expansion months, coverage is already at/near
nominal at **0 recession months** and rare-event count barely moves it. So the
calibration-score distribution is governed mainly by the *breadth/vintage* of
expansion months included (older history spans more diverse macro conditions,
producing a wider, more robust p90), with rare-event months a secondary
contributor.

**Recommended paper change.** Soften the central claim. The airtight,
fixed-size-supported statement is: *"ACI coverage under this rare-event shift is
governed by calibration-window **composition and diversity**, not raw size:
adding calibration months only helps insofar as they broaden the nonconformity-
score distribution (via rare-event episodes or a wider expansion vintage). At a
fixed size, coverage improves gradually — not as a sharp threshold — with
rare-event representation, most at the Current and 6M horizons."* The original
5-point sweep's abrupt 20→25% jump should be attributed to the joint effect of
(a) escaping the too-small 127-row window and (b) admitting the 2008 episode —
it conflates size and composition and should not be presented as isolating
composition.

---

## Compromises / honesty flags
- P0/P1/P2 all computed from the saved ensemble's in-sample training predictions
  (`preds_train_full`); **no model retraining**, per spec.
- A true **0-recession** N=254 variant WAS achievable (581 expansion months
  available) — no compromise needed there.
- **The main finding contradicts the pre-registered expectation**: P2 does *not*
  reproduce a sharp threshold at fixed size. Per the task's honesty standard,
  this is reported as-is; the "sharp threshold" framing needs softening to a
  gradual composition/diversity effect.
- P2 is sensitive to expansion-month selection; I ran 3 fills (recent/oldest/
  random) rather than one, so the sensitivity is documented rather than hidden.

## Files produced
- `fix-reg/mechanism_and_ablation_findings.md` (this file)
- P0: `nonconformity_by_regime_stats.csv`, `nonconformity_transition_20_25.csv`, `nonconformity_scores_raw.csv`, `fix-reg/task_p0_mechanism.py`
- P1: `figures/nonconformity_by_regime.pdf` + `.png`, `fix-reg/make_fig_nonconformity.py`
- P2: `fixed_size_composition_ablation.csv`, `fix-reg/task_p2_ablation.py`
