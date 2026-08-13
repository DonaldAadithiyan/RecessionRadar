"""
TASK 18, Items 1 / 3 / 4 / 5 / 7 — the recession-testbed review responses.

All five run on the primary testbed, out-of-fold, same protocol as every other
baseline in the paper. Grouped into one script because they share the same score
pools and test streams.

ITEM 1 — weighted / regime-weighted conformal baseline.
    Weight each pooled score by w_i ∝ exp(-lambda * d_i), d_i = months since the
    most recent matching-regime transition, and take the WEIGHTED empirical
    quantile as ACI's calibration quantile.
    lambda is FIXED IN ADVANCE from a structural rationale (half-life = the mean
    duration of a regime episode in the training data), never tuned on test.

ITEM 3 — Winkler interval score, selector vs DtACI.
    PRE-SPECIFIED BEFORE COMPUTING (spec guardrail): the Winkler score for a
    central (1-alpha) interval [l,u],
        W = (u-l) + (2/alpha)(l-y) if y<l ; + (2/alpha)(y-u) if y>u ; else (u-l)
    Lower is better. Chosen because it is the standard proper scoring rule for
    interval forecasts and penalises width and miscoverage on one scale.

ITEM 4 — EVT threshold sensitivity.
    Sweep the GPD threshold over {0.85, 0.90, 0.95} (plus 0.80 as the paper's
    current setting) and report coverage, width, and the fitted tail index xi at
    each — the standard GPD stability diagnostic.

ITEM 5 — recency/tail split ablation.
    Sweep the budget split 0/25/50/75/100% tail-vs-recency at fixed N, reporting
    Q_C, coverage AND width at each point. The review asked for the comparison
    "holding width constant"; width is reported explicitly so it is visible
    whether it actually is constant (it is not) rather than assumed.

ITEM 7 — quantile-convention sensitivity.
    Re-run the alpha-boundary ceiling sweep under nearest-rank (the proof's
    convention) and linear interpolation (NumPy's default) and compare. Verifies
    empirically that the q>0.5 boundary is not convention-dependent beyond an
    O(1/N) shift. Reported whatever it shows.

Outputs:
  fix-reg/task18_1_weighted_cp.csv
  fix-reg/task18_3_utility_scores.csv
  fix-reg/task18_4_evt_sensitivity.csv
  fix-reg/task18_5_recency_tail_split.csv
  fix-reg/task18_7_quantile_convention.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()

from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, wilson_ci_from_indicator, GAMMA_DEFAULT,
    ALPHA_TARGET,
)
import selector_lib as SEL  # noqa: E402
import baselines_lib as B  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
LABELS = ["Current", "1M", "3M", "6M"]

import task_oof_and_probit as T  # noqa: E402
REC_PROB = T.train_df["recession_probability"].values
IS_RARE = REC_PROB >= 50


def scores_for(h):
    return np.abs(T.oof_pred[:, h] - T.y_train[:, h])


print("=" * 96)
print("TASK 18 — Items 1, 3, 4, 5, 7 (primary recession testbed)")
print("=" * 96)


# =============================================================================
# ITEM 1 — weighted / regime-weighted conformal
# =============================================================================
print("\n" + "-" * 96)
print("ITEM 1 — regime-weighted conformal baseline")
print("-" * 96)

# lambda fixed IN ADVANCE from the training data's regime structure:
# half-life = mean duration of a rare-event episode. No test-set tuning.
transitions = np.where(np.diff(IS_RARE.astype(int)) != 0)[0]
episodes = []
if len(transitions):
    cur = IS_RARE[0]; start = 0
    for t in list(transitions) + [len(IS_RARE) - 1]:
        if IS_RARE[start] :
            episodes.append(t - start + 1)
        start = t + 1
mean_rare_dur = float(np.mean(episodes)) if episodes else 6.0
HALF_LIFE = max(mean_rare_dur, 1.0)
LAMBDA = np.log(2) / HALF_LIFE
print(f"  regime episodes in training: {len(episodes)}, "
      f"mean duration {mean_rare_dur:.1f} months")
print(f"  -> half-life = {HALF_LIFE:.1f} months, lambda = {LAMBDA:.4f} "
      f"(fixed before touching test data)")


def regime_distance(is_rare):
    """Months since the most recent regime transition, per pool index."""
    d = np.zeros(len(is_rare), dtype=float)
    last = 0
    for i in range(1, len(is_rare)):
        if is_rare[i] != is_rare[i - 1]:
            last = i
        d[i] = i - last
    return d


def weighted_quantile(values, weights, q):
    """Weighted empirical quantile (inverse-CDF convention)."""
    v = np.asarray(values, float); w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    if len(v) == 0:
        return np.nan
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(cw, np.clip(q, 0, 1))]
                 if np.searchsorted(cw, np.clip(q, 0, 1)) < len(v) else v[-1])


def run_weighted_aci(y_te, p_te, cal_scores, cal_weights,
                     gamma=GAMMA_DEFAULT, at=ALPHA_TARGET):
    a = at
    covered, widths = [], []
    for t in range(len(y_te)):
        q = weighted_quantile(cal_scores, cal_weights, np.clip(1 - a, 0, 1))
        if not np.isfinite(q):
            q = np.nanquantile(cal_scores, np.clip(1 - a, 0, 1))
        lo, hi = p_te[t] - q, p_te[t] + q
        widths.append(2 * q)
        yv = y_te[t]
        if np.isnan(yv):
            covered.append(np.nan); continue
        miss = 1 if (yv < lo or yv > hi) else 0
        covered.append(1 - miss)
        a = float(np.clip(a + gamma * (at - miss), 0.01, 0.99))
    return np.array(covered), np.array(widths)


dist = regime_distance(IS_RARE)
i1_rows = []
for h_idx, h in enumerate(LABELS):
    s = scores_for(h_idx)
    valid = np.where(np.isfinite(s))[0]
    y_te, p_te = T.y_test[:, h_idx], T.preds_test[:, h_idx]

    # reference points
    trailing = s[valid[-N_FIX:]]
    cov_b, _, w_b = run_aci(y_te, p_te, trailing, gamma=GAMMA_DEFAULT)
    cb, wb = coverage_and_width(cov_b, w_b)
    sel = s[SEL.support_width_selector(s, N_FIX)]
    cov_s, _, w_s = run_aci(y_te, p_te, sel[np.isfinite(sel)],
                            gamma=GAMMA_DEFAULT)
    cs, ws = coverage_and_width(cov_s, w_s)

    # regime-weighted over the FULL pool (that is the method's premise:
    # use every score, weighted, rather than selecting a subset)
    wgt = np.exp(-LAMBDA * dist[valid])
    cov_w, w_w = run_weighted_aci(y_te, p_te, s[valid], wgt)
    cw, ww = coverage_and_width(cov_w, w_w)
    lo, hi = wilson_ci_from_indicator(cov_w)

    i1_rows.append(dict(horizon=h, lambda_=round(LAMBDA, 4),
                        half_life=round(HALF_LIFE, 1),
                        trailing_cov=round(cb, 2), trailing_w=round(wb, 2),
                        weighted_cov=round(cw, 2), weighted_w=round(ww, 2),
                        weighted_lo=round(lo, 2), weighted_hi=round(hi, 2),
                        selector_cov=round(cs, 2), selector_w=round(ws, 2),
                        beats_trailing=bool(cw > cb),
                        closes_gap_frac=round((cw - cb) / (cs - cb), 3)
                        if abs(cs - cb) > 1e-9 else None))
    r = i1_rows[-1]
    print(f"  {h:8s} trailing={r['trailing_cov']:6.2f}  "
          f"weighted={r['weighted_cov']:6.2f} (w={r['weighted_w']:7.2f})  "
          f"selector={r['selector_cov']:6.2f}  "
          f"closes {r['closes_gap_frac']}")

pd.DataFrame(i1_rows).to_csv(f"{OUT}/task18_1_weighted_cp.csv", index=False)


# =============================================================================
# ITEM 3 — Winkler interval score (pre-specified above)
# =============================================================================
print("\n" + "-" * 96)
print("ITEM 3 — Winkler interval score: selector vs DtACI (lower is better)")
print("-" * 96)


def winkler(y, lo, hi, alpha=ALPHA_TARGET):
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


def aci_intervals(y_te, p_te, cal, gamma=GAMMA_DEFAULT, at=ALPHA_TARGET):
    """Re-run ACI, returning the interval bounds so a scoring rule can be applied."""
    a = at
    los, his = [], []
    for t in range(len(y_te)):
        q = np.quantile(cal, np.clip(1 - a, 0, 1))
        los.append(p_te[t] - q); his.append(p_te[t] + q)
        yv = y_te[t]
        if np.isnan(yv):
            continue
        miss = 1 if (yv < p_te[t] - q or yv > p_te[t] + q) else 0
        a = float(np.clip(a + gamma * (at - miss), 0.01, 0.99))
    return np.array(los), np.array(his)


i3_rows = []
for h_idx, h in enumerate(LABELS):
    s = scores_for(h_idx)
    valid = np.where(np.isfinite(s))[0]
    y_te, p_te = T.y_test[:, h_idx], T.preds_test[:, h_idx]
    trailing = s[valid[-N_FIX:]]
    sel = s[SEL.support_width_selector(s, N_FIX)]
    sel = sel[np.isfinite(sel)]

    entries = {}
    lo_s, hi_s = aci_intervals(y_te, p_te, sel)
    entries["diversity_optimal"] = (lo_s, hi_s)
    lo_b, hi_b = aci_intervals(y_te, p_te, trailing)
    entries["pooled_trailing"] = (lo_b, hi_b)

    # DtACI produces widths directly; reconstruct bounds from them
    cov_d, wid_d = B.run_dtaci(y_te, p_te, trailing)
    lo_d = p_te - np.asarray(wid_d) / 2.0
    hi_d = p_te + np.asarray(wid_d) / 2.0
    entries["dtaci"] = (lo_d, hi_d)

    for name, (lo, hi) in entries.items():
        sc = [winkler(y_te[t], lo[t], hi[t])
              for t in range(len(y_te)) if np.isfinite(y_te[t])]
        i3_rows.append(dict(horizon=h, strategy=name, n=len(sc),
                            winkler_mean=round(float(np.mean(sc)), 3),
                            winkler_median=round(float(np.median(sc)), 3)))
    sub = [r for r in i3_rows if r["horizon"] == h]
    best = min(sub, key=lambda r: r["winkler_mean"])
    print(f"  {h:8s} " + "  ".join(
        f"{r['strategy']}={r['winkler_mean']:8.2f}" for r in sub)
        + f"   -> best: {best['strategy']}")

pd.DataFrame(i3_rows).to_csv(f"{OUT}/task18_3_utility_scores.csv", index=False)


# =============================================================================
# ITEM 4 — EVT threshold sensitivity
# =============================================================================
print("\n" + "-" * 96)
print("ITEM 4 — EVT threshold sensitivity + tail-index stability")
print("-" * 96)

i4_rows = []
for h_idx, h in enumerate(LABELS):
    s = scores_for(h_idx)
    valid = np.where(np.isfinite(s))[0]
    trailing = s[valid[-N_FIX:]]
    y_te, p_te = T.y_test[:, h_idx], T.preds_test[:, h_idx]
    for thr in [0.80, 0.85, 0.90, 0.95]:
        fit = B.fit_tail(trailing, threshold_q=thr) if hasattr(B, "fit_tail") \
            else None
        if fit is None:
            u = float(np.quantile(trailing, thr))
            exc = trailing[trailing > u] - u
            xi = sigma = np.nan
            if len(exc) >= 10:
                xi, _, sigma = st.genpareto.fit(exc, floc=0)
            n_exc = int(len(exc))
        else:
            u, xi, sigma, n_exc = fit

        # run EVT-tail ACI at this threshold
        def evt_q(cs, alpha, thr=thr):
            uu = np.quantile(cs, thr)
            ex = cs[cs > uu] - uu
            if len(ex) < 10:
                return np.quantile(cs, 1 - alpha)
            try:
                c, _, sc = st.genpareto.fit(ex, floc=0)
                pe = len(ex) / len(cs)
                tgt = alpha / pe
                if tgt >= 1 or tgt <= 0:
                    return np.quantile(cs, 1 - alpha)
                qq = uu + st.genpareto.ppf(1 - tgt, c, loc=0, scale=sc)
                return float(qq) if np.isfinite(qq) else np.quantile(cs, 1 - alpha)
            except Exception:
                return np.quantile(cs, 1 - alpha)

        a = ALPHA_TARGET
        covered, widths = [], []
        for t in range(len(y_te)):
            q = evt_q(trailing, np.clip(a, 0.001, 0.5))
            widths.append(2 * q)
            yv = y_te[t]
            if np.isnan(yv):
                covered.append(np.nan); continue
            miss = 1 if (yv < p_te[t] - q or yv > p_te[t] + q) else 0
            covered.append(1 - miss)
            a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss),
                              0.01, 0.99))
        cov, w = coverage_and_width(np.array(covered), np.array(widths))
        i4_rows.append(dict(horizon=h, threshold_q=thr, n_exceedances=n_exc,
                            xi=round(float(xi), 4) if np.isfinite(xi) else None,
                            sigma=round(float(sigma), 4) if np.isfinite(sigma) else None,
                            coverage=round(cov, 2), width=round(w, 3)))
    sub = [r for r in i4_rows if r["horizon"] == h]
    print(f"  {h:8s} " + "  ".join(
        f"q{r['threshold_q']}: xi={r['xi']}, cov={r['coverage']:.1f}"
        for r in sub))

I4 = pd.DataFrame(i4_rows)
I4.to_csv(f"{OUT}/task18_4_evt_sensitivity.csv", index=False)
print("\n  tail-index stability (xi should be roughly flat if the GPD fits):")
print(I4.pivot_table(index="horizon", columns="threshold_q",
                     values="xi").round(3).to_string())


# =============================================================================
# ITEM 5 — recency / tail split ablation, width visible
# =============================================================================
print("\n" + "-" * 96)
print("ITEM 5 — recency/tail split at fixed N (width reported, not assumed)")
print("-" * 96)


def split_select(scores, N, tail_frac):
    """tail_frac of the budget from the score tails, the rest from recency."""
    v = np.where(np.isfinite(scores))[0]
    n_tail = int(round(tail_frac * N))
    n_recent = N - n_tail
    recent = v[-n_recent:] if n_recent > 0 else np.array([], dtype=int)
    rest = np.setdiff1d(v, recent)
    if n_tail == 0 or len(rest) == 0:
        return np.sort(recent)
    order = rest[np.argsort(scores[rest])]
    chosen, lo, hi, take_low = [], 0, len(order) - 1, True
    while len(chosen) < min(n_tail, len(order)) and lo <= hi:
        if take_low:
            chosen.append(order[lo]); lo += 1
        else:
            chosen.append(order[hi]); hi -= 1
        take_low = not take_low
    return np.sort(np.concatenate([recent, np.array(chosen, dtype=int)]))


i5_rows = []
for h_idx, h in enumerate(LABELS):
    s = scores_for(h_idx)
    y_te, p_te = T.y_test[:, h_idx], T.preds_test[:, h_idx]
    for frac in [0.0, 0.25, 0.50, 0.75, 1.0]:
        idx = split_select(s, N_FIX, frac)
        cal = s[idx]; cal = cal[np.isfinite(cal)]
        covered, _, widths = run_aci(y_te, p_te, cal, gamma=GAMMA_DEFAULT)
        cov, w = coverage_and_width(covered, widths)
        lo, hi = wilson_ci_from_indicator(covered)
        i5_rows.append(dict(horizon=h, tail_frac=frac, n_cal=len(cal),
                            Q_C=round(float(np.quantile(cal, 1 - ALPHA_TARGET)), 3),
                            coverage=round(cov, 2),
                            wilson_lo=round(lo, 2), wilson_hi=round(hi, 2),
                            width=round(w, 3)))
    sub = [r for r in i5_rows if r["horizon"] == h]
    print(f"  {h:8s} " + "  ".join(
        f"{int(100*r['tail_frac'])}%:cov={r['coverage']:.1f},w={r['width']:.1f}"
        for r in sub))

I5 = pd.DataFrame(i5_rows)
I5.to_csv(f"{OUT}/task18_5_recency_tail_split.csv", index=False)
print("\n  WIDTH across splits (is it actually constant? the review assumed so):")
print(I5.pivot_table(index="horizon", columns="tail_frac",
                     values="width").round(2).to_string())


# =============================================================================
# ITEM 7 — quantile-convention sensitivity
# =============================================================================
print("\n" + "-" * 96)
print("ITEM 7 — ceiling ratio under nearest-rank vs linear interpolation")
print("-" * 96)

ALPHA_GRID = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.49, 0.50, 0.55, 0.60]


def q_nearest_rank(a, q):
    """Proof's convention: value at rank ceil(q*n), 1-indexed."""
    v = np.sort(np.asarray(a, float))
    n = len(v)
    if n == 0:
        return np.nan
    k = int(np.ceil(np.clip(q, 0, 1) * n))
    k = min(max(k, 1), n)
    return float(v[k - 1])


i7_rows = []
for h_idx, h in enumerate(LABELS):
    s = scores_for(h_idx)
    p = s[np.isfinite(s)]
    order = np.argsort(p)
    top = p[order[-N_FIX:]]
    sel = p[SEL.support_width_selector(p, N_FIX)]
    for a in ALPHA_GRID:
        q = 1 - a
        r_lin = float(np.quantile(sel, q) / np.quantile(top, q))
        nr_s, nr_t = q_nearest_rank(sel, q), q_nearest_rank(top, q)
        r_nr = float(nr_s / nr_t) if nr_t > 0 else np.nan
        i7_rows.append(dict(horizon=h, alpha=a, q=round(q, 3),
                            ratio_linear=round(r_lin, 6),
                            ratio_nearest_rank=round(r_nr, 6),
                            both_at_ceiling=bool(r_lin >= 1 - 1e-9 and
                                                 r_nr >= 1 - 1e-9),
                            conventions_agree=bool(
                                abs(r_lin - r_nr) < 1e-9)))
    sub = [r for r in i7_rows if r["horizon"] == h]
    print(f"  {h:8s} " + "  ".join(
        f"a{r['alpha']}:{r['ratio_nearest_rank']:.3f}/{r['ratio_linear']:.3f}"
        for r in sub if r["alpha"] in (0.10, 0.45, 0.49, 0.50, 0.55)))

I7 = pd.DataFrame(i7_rows)
I7.to_csv(f"{OUT}/task18_7_quantile_convention.csv", index=False)

sub05 = I7[I7["alpha"] < 0.5]
print(f"\n  alpha < 0.5: nearest-rank at ceiling in "
      f"{int((sub05['ratio_nearest_rank'] >= 1 - 1e-9).sum())}/{len(sub05)}, "
      f"linear at ceiling in "
      f"{int((sub05['ratio_linear'] >= 1 - 1e-9).sum())}/{len(sub05)}")
disagree = I7[~I7["conventions_agree"]]
print(f"  configurations where the two conventions disagree at all: "
      f"{len(disagree)} of {len(I7)}")
if len(disagree):
    print(disagree[["horizon", "alpha", "ratio_nearest_rank",
                    "ratio_linear"]].to_string(index=False))

print(f"\nSaved task18_1/3/4/5/7 CSVs to {OUT}/")
