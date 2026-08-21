"""
TASK 28, Items 4 & 5 — non-circular validation of q_alpha(z), and the full
frontier comparison against the expanded baseline set (now including RLCP,
added per Item 1's literature finding).

ITEM 4 (non-circular): q_alpha(z) is fit on a CAL-A subsplit; the per-bin
coverage check runs on CAL-B, DISJOINT from CAL-A. This is not a restatement of
the fitting objective -- a q_alpha that merely reproduces its training data will
fail on held-out bins.

ITEM 5: full frontier comparison. Reports coverage and width directly, not only
Winkler. Anti-gaming: win rate, vacuity ratio, block sign-flip permutation
(both domains temporal), BH across every cell.

RLCP (Hore & Barber 2023) is implemented as a kernel-weighted localized
conformal baseline, weighting calibration scores by exp(-gamma*||z_i - z_t||^2)
in the SAME 1-D coordinate errordir uses -- the fairest possible comparison,
isolating "learned monotone quantile function" vs "kernel reweighting" on
identical geometry.

Outputs: task28_4_bins.csv, task28_5_comparison.csv, task28_5_significance.csv
"""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg")); sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs; ensemble_stubs.install()
from domain_common import GAMMA_DEFAULT, ALPHA_TARGET, rolling_origin_folds
import selector_lib as SEL
import baselines_lib as B
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from task27_1_cqr import cqr_intervals
from task28_3_qalpha import fit_qalpha, window_size

OUT = "errordir"
LO, HI = 0.75, 1.25
FIT_FRAC, SEED = 0.60, 5
RLCP_GAMMA = 1.0        # bandwidth in standardized-z units; fixed in advance

_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    import domain_climate as CL
EN = pd.read_csv("data/domains/task25/energy.csv")


def get_domain(name):
    if name == "climate":
        return CL.X_pool, CL.y_pool, CL.X_test, np.asarray(CL.Y_TEST, float)
    y = EN["target"].values.astype(float)
    X = EN.drop(columns=["target"]).values.astype(float)
    keep = np.arange(max(0, len(y) - 4400), len(y))
    X, y = X[keep], y[keep]
    return X[:-400], y[:-400], X[-400:], y[-400:]


def winkler(y, lo, hi, alpha=ALPHA_TARGET):
    w = hi - lo
    if y < lo: return w + (2.0 / alpha) * (lo - y)
    if y > hi: return w + (2.0 / alpha) * (y - hi)
    return w


def rlcp_intervals(z_cal, s_cal, z_test, pred_test, gamma=RLCP_GAMMA,
                   alpha=ALPHA_TARGET):
    """Kernel-weighted localized conformal in the same 1-D coordinate."""
    sd = np.std(z_cal) + 1e-12
    zc, zt = z_cal / sd, z_test / sd
    los, his = [], []
    order = np.argsort(s_cal)
    s_sorted = s_cal[order]
    for t in range(len(zt)):
        w = np.exp(-gamma * (zc - zt[t]) ** 2)
        ws = w[order]
        cw = np.cumsum(ws) / max(ws.sum(), 1e-12)
        idx = int(np.searchsorted(cw, 1 - alpha))
        idx = min(max(idx, 0), len(s_sorted) - 1)
        q = float(s_sorted[idx])
        los.append(pred_test[t] - q); his.append(pred_test[t] + q)
    return np.array(los), np.array(his)


rows_bins, rows_cmp, wk_store = [], [], {}
for dom in ["climate", "energy"]:
    Xp, yp, Xt, yt = get_domain(dom)
    n_pool = len(yp); n_fit = int(round(FIT_FRAC * n_pool))
    FIT_I, CAL_I = np.arange(0, n_fit), np.arange(n_fit, n_pool)
    spread = float(np.percentile(yt, 95) - np.percentile(yt, 5))

    sc0 = StandardScaler().fit(Xp)
    okp = np.isfinite(yp)
    base_m = Ridge(alpha=1.0).fit(sc0.transform(Xp[okp]), yp[okp])
    oof = np.full(n_pool, np.nan)
    for tr, te in rolling_origin_folds(n_pool, n_folds=5):
        o = tr[np.isfinite(yp[tr])]
        if len(o) < 20: continue
        oof[te] = Ridge(alpha=1.0).fit(sc0.transform(Xp[o]), yp[o]).predict(sc0.transform(Xp[te]))
    scores = np.abs(oof - yp); ok = np.isfinite(scores)
    fit_ok, cal_ok = FIT_I[ok[FIT_I]], CAL_I[ok[CAL_I]]
    pred_test = base_m.predict(sc0.transform(Xt))
    cal_scores = scores[cal_ok]

    scb = StandardScaler().fit(Xp[fit_ok])
    bm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scb.transform(Xp[fit_ok]),
                                                    scores[fit_ok]).coef_
    bu = bm / (np.linalg.norm(bm) + 1e-12)
    z_cal = scb.transform(Xp[cal_ok]) @ bu
    z_test = scb.transform(Xt) @ bu

    # ── ITEM 4: fit q_alpha on CAL-A, validate per-bin on DISJOINT CAL-B ────
    rngs = np.random.default_rng(SEED)
    perm = rngs.permutation(len(cal_ok))
    a_idx, b_idx = np.sort(perm[:len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    qA, _, _ = fit_qalpha(z_cal[a_idx], cal_scores[a_idx])
    zb, sb = z_cal[b_idx], cal_scores[b_idx]
    qb = qA(zb)
    edges = np.quantile(zb, [0, .2, .4, .6, .8, 1.0])
    print(f"\n  {dom}: ITEM 4 held-out per-bin coverage "
          f"(q_alpha fit on {len(a_idx)}, checked on {len(b_idx)} disjoint)")
    for i in range(5):
        m = (zb >= edges[i]) & (zb <= edges[i + 1] if i == 4 else zb < edges[i + 1])
        if m.sum() < 5: continue
        cov = 100 * float(np.mean(sb[m] <= qb[m]))
        rows_bins.append(dict(domain=dom, bin=i + 1, n=int(m.sum()),
                              z_lo=round(float(edges[i]), 3),
                              z_hi=round(float(edges[i + 1]), 3),
                              coverage=round(cov, 2),
                              mean_qalpha=round(float(np.mean(qb[m])), 3)))
        print(f"    bin {i+1}: n={int(m.sum()):4d}  coverage={cov:6.2f}%  "
              f"mean q_alpha={np.mean(qb[m]):7.3f}")

    # ── ITEM 5: full comparison ────────────────────────────────────────────
    def aci(cs, mult=None):
        a = ALPHA_TARGET; los, his = [], []
        for t in range(len(yt)):
            q = float(np.quantile(cs, np.clip(1 - a, 0, 1)))
            if mult is not None: q *= float(mult[t])
            los.append(pred_test[t] - q); his.append(pred_test[t] + q)
            miss = 1 if (yt[t] < los[-1] or yt[t] > his[-1]) else 0
            a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss), .01, .99))
        return np.array(los), np.array(his)

    N = min(254, len(cal_scores)); trailing = cal_scores[-N:]
    arms = {"pooled_trailing": aci(trailing),
            "diversity_optimal": aci(cal_scores[SEL.support_width_selector(cal_scores, N)])}
    rare = cal_scores >= np.percentile(cal_scores, 90)
    cbr = {True: cal_scores[rare], False: cal_scores[~rare]}
    if min(len(cbr[True]), len(cbr[False])) >= 2:
        treg = [bool(x) for x in (yt >= np.percentile(yt, 90))]
        c, w = B.run_mondrian(yt, pred_test, cbr, treg)
        arms["mondrian"] = (pred_test - w / 2, pred_test + w / 2)
    for nm, fn, arg in [("pid_conformal", B.run_pid, trailing),
                        ("evt_tail", B.run_evt, cal_scores),
                        ("dtaci", B.run_dtaci, cal_scores),
                        ("acmcp", B.run_acmcp, cal_scores),
                        ("bellman_ci", B.run_bci, cal_scores)]:
        c, w = fn(yt, pred_test, arg)
        arms[nm] = (pred_test - w / 2, pred_test + w / 2)
    arms["cqr"] = cqr_intervals(Xp[fit_ok], yp[fit_ok], Xp[cal_ok], yp[cal_ok], Xt)
    arms["rlcp"] = rlcp_intervals(z_cal, cal_scores, z_test, pred_test)

    # existing fixed-multiplier errordir
    u = np.empty(len(z_test))
    for t in range(len(z_test)):
        ref = np.sort(np.concatenate([z_cal, z_test[:t]]))
        u[t] = np.searchsorted(ref, z_test[t], side="right") / max(len(ref), 1)
    arms["errordir_fixed"] = aci(cal_scores, mult=LO + (HI - LO) * np.clip(u, 0, 1))

    # NEW: q_alpha(z) directly supplies the half-width (full CAL)
    qFull, _, _ = fit_qalpha(z_cal, cal_scores)
    hw = qFull(z_test)
    arms["errordir_qalpha"] = (pred_test - hw, pred_test + hw)

    for nm, (los, his) in arms.items():
        wk = np.array([winkler(yt[t], los[t], his[t]) for t in range(len(yt))])
        cov = 100 * float(np.mean([(los[t] <= yt[t] <= his[t]) for t in range(len(yt))]))
        w = float(np.mean(his - los))
        rows_cmp.append(dict(domain=dom, method=nm, n=len(yt),
                             coverage=round(cov, 2), mean_width=round(w, 3),
                             width_spread_ratio=round(w / spread, 2),
                             vacuous=bool(w / spread > 100),
                             winkler_mean=round(float(np.mean(wk)), 3),
                             winkler_median=round(float(np.median(wk)), 3)))
        wk_store[(dom, nm)] = wk

pd.DataFrame(rows_bins).to_csv(f"{OUT}/task28_4_bins.csv", index=False)
C = pd.DataFrame(rows_cmp); C.to_csv(f"{OUT}/task28_5_comparison.csv", index=False)
for dom in C.domain.unique():
    print(f"\n{dom.upper()} — frontier (sorted by Winkler):")
    print(C[C.domain == dom].sort_values("winkler_mean")[
        ["method", "coverage", "mean_width", "width_spread_ratio",
         "winkler_mean", "winkler_median"]].to_string(index=False))

rng = np.random.default_rng(28)
srows = []
for dom in C.domain.unique():
    for tag in ["errordir_qalpha", "errordir_fixed"]:
        a = wk_store[(dom, tag)]
        for nm in ["pooled_trailing", "diversity_optimal", "mondrian", "pid_conformal",
                   "evt_tail", "dtaci", "acmcp", "bellman_ci", "cqr", "rlcp"]:
            if (dom, nm) not in wk_store: continue
            b = wk_store[(dom, nm)]
            n = min(len(a), len(b)); dd = a[:n] - b[:n]
            obs = float(np.mean(dd)); wr = 100 * float((dd < 0).mean())
            blocks = max(1, n // 12)
            nullb = np.array([float(np.mean(dd * np.repeat(
                rng.choice([-1, 1], size=blocks + 1), 12)[:n])) for _ in range(2000)])
            assert nullb.std() > 1e-12
            srows.append(dict(domain=dom, method=tag, baseline=nm, n=n,
                              mean_winkler_diff=round(obs, 3),
                              win_rate_pct=round(wr, 1),
                              perm_signflip_p=round(float((nullb <= obs).mean()), 4)))
S = pd.DataFrame(srows)
p = S["perm_signflip_p"].values
o = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
for rk in range(m - 1, -1, -1):
    i = o[rk]; prev = min(prev, p[i] * m / (rk + 1)); bh[i] = prev
S["perm_signflip_p_bh"] = np.round(bh, 4)
S.to_csv(f"{OUT}/task28_5_significance.csv", index=False)
print(f"\nBH family m={m}")
print(S[S.method == "errordir_qalpha"].to_string(index=False))
print(f"\nSaved task28_4_bins.csv, task28_5_comparison.csv, task28_5_significance.csv")
