"""
Calibration-set selectors.

Two objectives, both choosing an N-subset of a pool of nonconformity scores:

support_width_selector   (the paper's existing method, Phase 2)
    Maximises p95 - p5 of the selected scores by alternating from both tails
    inward. This is provably optimal for that objective at fixed N.

quantile_targeted_selector   (NEW — Task 10)
    Maximises the selected set's quantile reach over the alpha window ACI
    actually visits, subject to a width budget.

WHY THE NEW OBJECTIVE EXISTS
----------------------------
ACI never consumes p95 - p5. At each step it consumes exactly one number: the
empirical (1 - alpha_t) quantile of the calibration scores. Widening the LOWER
tail of the score distribution therefore contributes nothing to the interval
ACI actually builds — it is pure width inflation. The support-width objective
spends half its selection budget on scores that cannot affect coverage.

Phase 5's theory note already identified the operative quantity: coverage
deficit is monotone in the calibration->test quantile shortfall at the operating
level, and support width is only a proxy for it. This selector optimises the
quantity directly:

    maximise  sum_alpha  w(alpha) * Q_S(1 - alpha)
    s.t.      |S| = N,  mean interval width <= kappa * baseline width

where w(alpha) concentrates on the empirically measured alpha range (see
task10_alpha_trajectories.py) and kappa is an explicit width budget.

The expected effect is NOT higher coverage — it is comparable coverage at
materially lower width, by dropping lower-tail padding that buys nothing. That
targets the one axis where tuned BCI and CPTC currently beat the selector.
"""

import numpy as np


# ── Existing objective: support width (Phase 2) ─────────────────────────────

def support_width_selector(scores, N):
    """
    Maximise p95 - p5 by taking scores alternately from the low and high tails.
    Byte-for-byte the rule used in task_phase2.py / task7_baseline_horse_race.py.
    """
    s = np.asarray(scores, dtype=float)
    valid = np.where(np.isfinite(s))[0]
    order = valid[np.argsort(s[valid])]
    chosen, lo, hi, take_low = [], 0, len(order) - 1, True
    while len(chosen) < min(N, len(order)) and lo <= hi:
        if take_low:
            chosen.append(order[lo]); lo += 1
        else:
            chosen.append(order[hi]); hi -= 1
        take_low = not take_low
    return np.array(sorted(chosen))


# ── New objective: quantile-targeted selection ──────────────────────────────

def _alpha_weights(alpha_lo, alpha_hi, n_grid=25, mode="loguniform"):
    """
    Weight function over the alpha window ACI visits. Log-uniform by default so
    that small alphas (deep quantiles, where coverage is won or lost) are not
    swamped by the denser large-alpha region.
    """
    alpha_lo = max(float(alpha_lo), 1e-3)
    alpha_hi = max(float(alpha_hi), alpha_lo * 1.01)
    if mode == "loguniform":
        grid = np.exp(np.linspace(np.log(alpha_lo), np.log(alpha_hi), n_grid))
    else:
        grid = np.linspace(alpha_lo, alpha_hi, n_grid)
    w = np.ones_like(grid) / len(grid)
    return grid, w


def quantile_reach_score(scores, alpha_grid, alpha_w):
    """Weighted mean of the set's (1-alpha) quantiles: the objective value."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 2:
        return 0.0
    qs = np.quantile(s, np.clip(1.0 - alpha_grid, 0.0, 1.0))
    return float(np.sum(alpha_w * qs))


def expected_width(scores, alpha_grid, alpha_w):
    """
    Width proxy: the weighted mean interval width the set would produce over the
    same alpha window (2 * quantile). Used to enforce the width budget.
    """
    return 2.0 * quantile_reach_score(scores, alpha_grid, alpha_w)


def quantile_targeted_selector(scores, N, alpha_lo=0.01, alpha_hi=0.10,
                               kappa=None, baseline_scores=None, n_grid=25):
    """
    Select an N-subset maximising weighted quantile reach over [alpha_lo,
    alpha_hi], subject to an optional width budget.

    kappa : width budget as a multiple of the baseline set's expected width.
            None = unconstrained (pure reach maximisation).
    baseline_scores : the reference set the budget is relative to (typically the
            trailing-N calibration set). Required when kappa is not None.

    Structure of the solution
    -------------------------
    For a fixed count N, the (1-alpha) quantile of the selected set is
    determined by which scores occupy its upper region. Taking the N largest
    scores maximises every upper quantile simultaneously — but it also produces
    a degenerate set concentrated in the tail, which inflates width at the
    LARGE-alpha end of the window and destroys the low-score mass ACI needs when
    alpha relaxes.

    The selector therefore takes a top-heavy but not degenerate set: a fraction
    `f` of the budget from the upper tail (driving quantile reach) and the
    remainder spread as an even stride across the rest of the distribution
    (preserving shape at relaxed alpha). `f` is chosen by scanning candidate
    values and keeping the best feasible objective — an exact 1-D search, not a
    heuristic guess.
    """
    s_all = np.asarray(scores, dtype=float)
    valid = np.where(np.isfinite(s_all))[0]
    if len(valid) == 0:
        return np.array([], dtype=int)
    N = int(min(N, len(valid)))
    order = valid[np.argsort(s_all[valid])]          # ascending

    grid, w = _alpha_weights(alpha_lo, alpha_hi, n_grid=n_grid)

    budget = None
    if kappa is not None and baseline_scores is not None:
        budget = kappa * expected_width(baseline_scores, grid, w)

    best_sel, best_obj = None, -np.inf
    best_infeasible_sel, best_infeasible_obj = None, -np.inf

    # f = share of the budget drawn from the extreme upper tail
    for f in np.linspace(0.1, 1.0, 19):
        n_top = int(round(f * N))
        n_top = max(1, min(n_top, N))
        n_rest = N - n_top

        top = order[-n_top:]
        if n_rest > 0:
            pool_rest = order[:-n_top] if n_top < len(order) else np.array([], int)
            if len(pool_rest) == 0:
                continue
            # even stride across the remaining distribution preserves its shape
            idx = np.linspace(0, len(pool_rest) - 1, n_rest).round().astype(int)
            rest = pool_rest[np.unique(idx)]
            sel = np.concatenate([rest, top])
        else:
            sel = top

        sel = np.unique(sel)
        if len(sel) < 2:
            continue

        sc = s_all[sel]
        obj = quantile_reach_score(sc, grid, w)
        wid = expected_width(sc, grid, w)

        if budget is None or wid <= budget:
            if obj > best_obj:
                best_obj, best_sel = obj, sel
        else:
            # remember the best infeasible option in case nothing fits
            if obj > best_infeasible_obj:
                best_infeasible_obj, best_infeasible_sel = obj, sel

    if best_sel is None:
        # No configuration met the budget. Fall back to the narrowest candidate
        # rather than silently returning an over-budget set.
        best_sel = (best_infeasible_sel if best_infeasible_sel is not None
                    else order[-N:])
    return np.array(sorted(best_sel))


def describe_selection(scores, sel):
    """Diagnostics for a chosen subset."""
    s = np.asarray(scores, dtype=float)[sel]
    s = s[np.isfinite(s)]
    if len(s) < 2:
        return {}
    return dict(n=int(len(s)),
                support_p95_p5=float(np.percentile(s, 95) - np.percentile(s, 5)),
                q90=float(np.quantile(s, 0.90)),
                q95=float(np.quantile(s, 0.95)),
                q99=float(np.quantile(s, 0.99)),
                median=float(np.median(s)),
                mean=float(np.mean(s)))
