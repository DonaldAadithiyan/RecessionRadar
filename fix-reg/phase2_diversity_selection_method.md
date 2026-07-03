# Phase 2 — A diversity-aware calibration selection method

**Verdict: the diversity-maximizing selector improves coverage at all 4 horizons
and — uniquely among every strategy tried in this paper — meaningfully lifts 6M
coverage (67.8 → 81.4%). But it buys coverage with much wider intervals, and 6M
still falls short of nominal, so the honest claim is "diversity-optimal selection
is necessary but not sufficient at long horizons."**

## Algorithm (greedy, per horizon)

Budget N=254 from the 635-month pool. Objective: maximize **support width
(p95−p5)** of the calibration set's pooled nonconformity scores (the Phase-1
winner). Because support width is a two-tail spread, the greedy maximizer is
simple and near-exact: seed with the min- and max-score months, then add
remaining months alternating from the low and high tails inward until N=254.
(For a fixed score set and fixed count, taking the extreme-tail months provably
maximizes p95−p5; greedy = optimal here, so no exact solver was needed.) Run per
horizon since scores are horizon-specific. Code: `fix-reg/task_phase2.py`.

## Comparison (same test set, same ACI, N=254)

| Horizon | Trailing-40% cov | Fixed-abl (16 rare) cov | **Diversity-opt cov** | Trail. w | Div-opt w |
|---|---|---|---|---|---|
| Current | 89.23 | 89.23 | **92.31** | 7.45 | 13.16 |
| 1M | 85.94 | 85.94 | **89.06** | 7.35 | 21.13 |
| 3M | 85.48 | 85.48 | **90.32** | 14.95 | 25.35 |
| 6M | 67.80 | 67.80 | **81.36** | 10.64 | 32.58 |

Diversity-optimal support widths: 10.7 / 14.3 / 18.9 / 21.7 (vs. trailing 4.0 /
4.3 / 10.1 / 3.5) — the selector roughly triples–sextuples the score support,
and coverage rises accordingly, confirming the Phase-1 mechanism *prospectively*.

## The 6-month question (answered honestly)

Diversity-optimal selection is the **first strategy in this paper to move 6M
coverage at all** — from a stuck ~66–68% (every prior trailing/ablation setting)
to **81.4%**, a +13.6pp gain. It does **not** reach the 90% nominal, so:
> "Diversity-optimal calibration selection is *necessary but not sufficient* for
> long-horizon coverage: it closes roughly half the 6-month coverage gap
> (67.8→81.4% toward 90%) where all size- and composition-based strategies closed
> none, but full nominal coverage at 6 months remains out of reach — consistent
> with the 6-month forecaster's residuals being not merely narrow but structurally
> mis-scaled during the shift."

## Honest caveats
- **Cost is width.** Coverage is bought with substantially wider intervals (6M:
  10.6→32.6 pp). This is the expected sharpness–coverage tradeoff and must be
  reported alongside coverage; diversity-optimal is not a free lunch.
- **Diversity-optimal ≈ rare-heavy at this pool.** Maximizing score support
  naturally recruits 41–49 of the 50 rare-event months (they *are* the
  high-score months), so here the diversity-optimal set overlaps heavily with a
  rare-event-maximizing set. This is consistent with — not contradictory to —
  the paper's thesis: rare months help *because* they widen the score
  distribution, and Phase 1/4a already showed diversity predicts coverage even
  when rare-count is held fixed. But we should not claim the two are cleanly
  separable at selection time on this particular series.
- Selection uses in-sample training residuals from the saved ensemble; no
  retraining. A deployment version would need out-of-fold calibration scores.

Data: `fix-reg/phase2_selection_comparison.csv`.
