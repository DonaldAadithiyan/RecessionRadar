"""
TASK 29, Item 2 — the decisive comparison: ORIGINAL errordir vs RLCP.

errordir configuration = the one that actually won: beta_mean, rank-based fixed
multiplier [0.75, 1.25], rolling recentering. NOT the q_alpha(z) redesign.

Identical calibration-split discipline for both methods (Task 24 Finding 3):
both use the same FIT split for beta / bandwidth, the same CAL split for
calibration scores, and the same TEST split. Neither sees data the other is
denied.

Also runs baseLCP (Task 28's construction) to (a) verify this harness reproduces
Task 28's reported RLCP numbers and (b) show what the randomization + (n+1)
normalisation actually change.

RLCP bandwidth: median heuristic on FIT projections only, plus a sensitivity
band (gamma x {0.25, 0.5, 1, 2, 4}) so RLCP is not disadvantaged by one
bandwidth choice. The BEST RLCP over that band is reported alongside the
default -- deliberately generous to RLCP, since the burden here is on errordir.

Anti-gaming: win rate, vacuity ratio, block sign-flip permutation (both domains
temporal), BH across every cell.

Outputs: task29_1_rlcp_verify.csv, task29_2_comparison.csv,
         task29_2_significance.csv
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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from task29_1_rlcp import rlcp_intervals, baselcp_intervals, median_heuristic_gamma

OUT = "errordir"
LO, HI = 0.75, 1.25
FIT_FRAC, SEED = 0.60, 5
GAMMA_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]

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
    if not np.isfinite(hi - lo):
        return np.inf
    w = hi - lo
    if y < lo: return w + (2.0 / alpha) * (lo - y)
    if y > hi: return w + (2.0 / alpha) * (y - hi)
    return w


rows, wk_store, vrows = [], {}, []
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

    # beta_mean on FIT only — the winning configuration
    scb = StandardScaler().fit(Xp[fit_ok])
    bm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scb.transform(Xp[fit_ok]),
                                                    scores[fit_ok]).coef_
    bu = bm / (np.linalg.norm(bm) + 1e-12)
    z_fit = scb.transform(Xp[fit_ok]) @ bu
    z_cal = scb.transform(Xp[cal_ok]) @ bu
    z_test = scb.transform(Xt) @ bu

    def summarize(name, los, his, extra=None):
        fin = np.isfinite(his - los)
        wk = np.array([winkler(yt[t], los[t], his[t]) for t in range(len(yt))])
        cov = 100 * float(np.mean([(los[t] <= yt[t] <= his[t]) for t in range(len(yt))]))
        w = float(np.mean((his - los)[fin])) if fin.any() else np.inf
        rec = dict(domain=dom, method=name, n=len(yt),
                   coverage=round(cov, 2),
                   mean_width=round(w, 3) if np.isfinite(w) else None,
                   width_spread_ratio=round(w / spread, 2) if np.isfinite(w) else None,
                   vacuous=bool(np.isfinite(w) and w / spread > 100),
                   n_unbounded=int((~fin).sum()),
                   winkler_mean=round(float(np.mean(wk)), 3) if np.all(np.isfinite(wk)) else None,
                   winkler_median=round(float(np.median(wk)), 3))
        if extra: rec.update(extra)
        return rec, wk

    # ── ORIGINAL errordir: beta_mean + fixed rank multiplier ───────────────
    a = ALPHA_TARGET; los, his = [], []
    u = np.empty(len(z_test))
    for t in range(len(z_test)):
        ref = np.sort(np.concatenate([z_cal, z_test[:t]]))
        u[t] = np.searchsorted(ref, z_test[t], side="right") / max(len(ref), 1)
    mult = LO + (HI - LO) * np.clip(u, 0, 1)
    for t in range(len(yt)):
        q = float(np.quantile(cal_scores, np.clip(1 - a, 0, 1))) * float(mult[t])
        los.append(pred_test[t] - q); his.append(pred_test[t] + q)
        miss = 1 if (yt[t] < los[-1] or yt[t] > his[-1]) else 0
        a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss), .01, .99))
    r, wk = summarize("errordir", np.array(los), np.array(his))
    rows.append(r); wk_store[(dom, "errordir")] = wk

    # ── RLCP (published) at the median-heuristic bandwidth ─────────────────
    g0 = median_heuristic_gamma(z_fit)
    lo_r, hi_r, unb = rlcp_intervals(z_cal, cal_scores, z_test, pred_test, g0)
    r, wk = summarize("rlcp", lo_r, hi_r, extra=dict(gamma=round(g0, 5)))
    rows.append(r); wk_store[(dom, "rlcp")] = wk

    # ── RLCP bandwidth sensitivity (generous to RLCP) ──────────────────────
    best = None
    for m in GAMMA_MULTS:
        g = g0 * m
        l2, h2, u2 = rlcp_intervals(z_cal, cal_scores, z_test, pred_test, g)
        rr, wk2 = summarize(f"rlcp_g{m}", l2, h2, extra=dict(gamma=round(g, 5)))
        vrows.append(rr)
        if rr["winkler_mean"] is not None and (best is None or rr["winkler_mean"] < best[0]):
            best = (rr["winkler_mean"], m, g, wk2, l2, h2)
    if best:
        l2, h2 = best[4], best[5]
        r, wk = summarize("rlcp_best_gamma", l2, h2,
                          extra=dict(gamma=round(best[2], 5), gamma_mult=best[1]))
        rows.append(r); wk_store[(dom, "rlcp_best_gamma")] = wk

    # ── baseLCP: reproduces Task 28's construction ─────────────────────────
    lo_b, hi_b = baselcp_intervals(z_cal, cal_scores, z_test, pred_test, gamma=1.0)
    r, wk = summarize("baselcp_task28", lo_b, hi_b)
    rows.append(r); wk_store[(dom, "baselcp_task28")] = wk

C = pd.DataFrame(rows)
C.to_csv(f"{OUT}/task29_2_comparison.csv", index=False)
pd.DataFrame(vrows).to_csv(f"{OUT}/task29_1_rlcp_verify.csv", index=False)
for dom in C.domain.unique():
    print(f"\n{dom.upper()}:")
    print(C[C.domain == dom][["method", "coverage", "mean_width",
                              "width_spread_ratio", "n_unbounded",
                              "winkler_mean", "winkler_median"]].to_string(index=False))

rng = np.random.default_rng(29)
srows = []
for dom in C.domain.unique():
    a = wk_store[(dom, "errordir")]
    for nm in ["rlcp", "rlcp_best_gamma", "baselcp_task28"]:
        if (dom, nm) not in wk_store: continue
        b = wk_store[(dom, nm)]
        n = min(len(a), len(b)); dd = a[:n] - b[:n]
        if not np.all(np.isfinite(dd)):
            srows.append(dict(domain=dom, baseline=nm, n=n,
                              mean_winkler_diff=None, win_rate_pct=None,
                              perm_signflip_p=None,
                              note="baseline produced unbounded intervals"))
            continue
        obs = float(np.mean(dd)); wr = 100 * float((dd < 0).mean())
        blocks = max(1, n // 12)
        nullb = np.array([float(np.mean(dd * np.repeat(
            rng.choice([-1, 1], size=blocks + 1), 12)[:n])) for _ in range(2000)])
        assert nullb.std() > 1e-12
        srows.append(dict(domain=dom, baseline=nm, n=n,
                          mean_winkler_diff=round(obs, 3),
                          win_rate_pct=round(wr, 1),
                          perm_signflip_p=round(float((nullb <= obs).mean()), 4),
                          note=""))
S = pd.DataFrame(srows)
fin = S["perm_signflip_p"].notna()
p = S.loc[fin, "perm_signflip_p"].values
o = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
for rk in range(m - 1, -1, -1):
    i = o[rk]; prev = min(prev, p[i] * m / (rk + 1)); bh[i] = prev
S.loc[fin, "perm_signflip_p_bh"] = np.round(bh, 4)
S.to_csv(f"{OUT}/task29_2_significance.csv", index=False)
print("\n" + "=" * 96)
print("SIGNIFICANCE — errordir vs RLCP (negative diff = errordir better)")
print("=" * 96)
print(S.to_string(index=False))
print(f"\nSaved task29_2_comparison.csv, task29_2_significance.csv")
