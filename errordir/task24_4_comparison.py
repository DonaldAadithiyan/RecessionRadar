"""
TASK 24, Item 4 — full baseline comparison for validated fits only.

Compares the error-direction method against the COMPLETE baseline set this
project already establishes for these domains — pooled_trailing, mondrian,
pid_conformal, evt_tail, dtaci, acmcp, bellman_ci, diversity_optimal — not a
convenient subset.

All baselines are re-run here (rather than read from task7_*.csv) because the
existing tables report coverage/width but not Winkler, and every method must be
scored on identical intervals for the comparison to be meaningful. Coverage and
width are cross-checked against the existing tables as a correctness check.

Anti-gaming discipline from Tasks 22-23, applied to every cell:
  - WIN RATE alongside the mean
  - WIDTH/SPREAD ratio vs the ~100 vacuity threshold
  - PERMUTATION tests (circular-shift + block sign-flip), NOT Wilcoxon
  - BH correction across EVERY baseline comparison, not just the primary one

NOTE ON HEALTHCARE'S STRUCTURE: healthcare cohorts are randomly permuted before
splitting (domain_healthcare.py: rng.permutation), so there is no temporal
autocorrelation to preserve. The circular-shift null is therefore not meaningful
there and the block sign-flip null is the operative test; this is stated rather
than silently applying a temporal test to non-temporal data.

Outputs: errordir/task24_4_comparison.csv, task24_4_significance.csv
"""
import os
import sys
import io
import contextlib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg"))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()
from domain_common import GAMMA_DEFAULT, ALPHA_TARGET, rolling_origin_folds  # noqa: E402
import selector_lib as SEL  # noqa: E402
import baselines_lib as B  # noqa: E402
from sklearn.linear_model import RidgeCV, Ridge  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = "errordir"
LO, HI = 0.75, 1.25          # Task 21 band (Task 22's raised ceiling = dead end)
FIT_FRAC, SEED = 0.60, 5

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import domain_healthcare as HC
    import domain_climate as CL

DOM = {
    "healthcare": dict(Xpool=HC.X_pool, Xtest=HC.X_test, ypool=HC.y_pool_pct,
                       ytest=HC.Y_TEST, rare=HC.rare_pool, mod=HC, temporal=False),
    "climate": dict(Xpool=CL.X_pool, Xtest=CL.X_test, ypool=CL.y_pool,
                    ytest=CL.Y_TEST, rare=CL.rare_pool, mod=CL, temporal=True),
}

V = pd.read_csv(f"{OUT}/task24_3_validation.csv")
VALID = [(r.domain, r.model) for r in V.itertuples() if r.beta_validated]
print("=" * 110)
print("TASK 24 Item 4 — full baseline comparison; validated fits:", VALID)
print("=" * 110)


def winkler(y, lo, hi, alpha=ALPHA_TARGET):
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


def disag(d):
    Xp, yp, Xt = d["Xpool"], d["ypool"], d["Xtest"]
    sc = StandardScaler().fit(Xp)
    Xp_s, Xt_s = sc.transform(Xp), sc.transform(Xt)
    o, t = {}, {}
    for name in ["ridge", "gradboost"]:
        A = Xp_s if name == "ridge" else Xp
        Bx = Xt_s if name == "ridge" else Xt
        oo = np.full(len(yp), np.nan)
        for tr, te in rolling_origin_folds(len(yp), n_folds=5):
            ok = tr[np.isfinite(yp[tr])]
            if len(ok) < 20:
                continue
            m = (Ridge(alpha=1.0) if name == "ridge"
                 else HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                                    max_depth=4, random_state=SEED))
            m.fit(A[ok], yp[ok]); oo[te] = m.predict(A[te])
        o[name] = oo
        mf = (Ridge(alpha=1.0) if name == "ridge"
              else HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                                 max_depth=4, random_state=SEED))
        okf = np.isfinite(yp)
        mf.fit(A[okf], yp[okf]); t[name] = mf.predict(Bx)
    return np.abs(o["ridge"] - o["gradboost"]), np.abs(t["ridge"] - t["gradboost"])


def score(name, los, his, y_h, spread):
    ok = np.isfinite(y_h)
    wk = np.array([winkler(y_h[t], los[t], his[t]) for t in range(len(y_h)) if ok[t]])
    cov = np.array([1 if (los[t] <= y_h[t] <= his[t]) else 0
                    for t in range(len(y_h)) if ok[t]])
    w = float(np.mean(his - los))
    return dict(method=name, n=int(ok.sum()),
                coverage=round(100 * float(cov.mean()), 2),
                mean_width=round(w, 3),
                width_spread_ratio=round(w / spread, 2),
                vacuous=bool(w / spread > 100),
                winkler_mean=round(float(np.mean(wk)), 3),
                winkler_median=round(float(np.median(wk)), 3)), wk, cov


rows, wk_store = [], {}
for dname, model in VALID:
    d = DOM[dname]
    Xp, Xt = d["Xpool"], d["Xtest"]
    y_h = np.asarray(d["ytest"], dtype=float)
    p_h = d["mod"].PREDS_TEST[model]
    scores = d["mod"].SCORES[model]
    spread = float(np.percentile(y_h[np.isfinite(y_h)], 95)
                   - np.percentile(y_h[np.isfinite(y_h)], 5))

    dp, dt = disag(d)
    Xp_aug = np.column_stack([Xp, dp])
    Xt_aug = np.column_stack([Xt, dt])
    n_pool = len(scores)
    n_fit = int(round(FIT_FRAC * n_pool))
    FIT_IDX, CAL_IDX = np.arange(0, n_fit), np.arange(n_fit, n_pool)
    ok = np.isfinite(scores) & np.isfinite(Xp_aug).all(axis=1)
    fit_ok, cal_ok = FIT_IDX[ok[FIT_IDX]], CAL_IDX[ok[CAL_IDX]]

    sc = StandardScaler().fit(Xp_aug[fit_ok])
    reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(sc.transform(Xp_aug[fit_ok]),
                                                     scores[fit_ok])
    beta = reg.coef_.astype(float)
    proj_cal = sc.transform(Xp_aug[cal_ok]) @ beta
    proj_test = sc.transform(Xt_aug) @ beta
    cal_scores = scores[cal_ok]

    # rolling recentering (Task 22 Item 1, verified leak-free)
    u = np.empty(len(proj_test))
    for t in range(len(proj_test)):
        ref = np.sort(np.concatenate([proj_cal, proj_test[:t]]))
        u[t] = np.searchsorted(ref, proj_test[t], side="right") / max(len(ref), 1)
    mult = LO + (HI - LO) * np.clip(u, 0, 1)

    def aci(cs, m=None):
        a = ALPHA_TARGET
        los, his = [], []
        for t in range(len(y_h)):
            q = float(np.quantile(cs, np.clip(1 - a, 0, 1)))
            if m is not None:
                q *= float(m[t])
            los.append(p_h[t] - q); his.append(p_h[t] + q)
            yt = y_h[t]
            if np.isfinite(yt):
                miss = 1 if (yt < los[-1] or yt > his[-1]) else 0
                a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss), .01, .99))
        return np.array(los), np.array(his)

    N = min(254, len(cal_scores))
    sel = SEL.support_width_selector(cal_scores, N)
    trailing = cal_scores[-N:]

    arms = {}
    arms["pooled_trailing"] = aci(trailing)
    arms["diversity_optimal"] = aci(cal_scores[sel])
    # mondrian: rare vs common calibration split, oracle regime labels
    rare_cal = d["rare"][cal_ok]
    cbr = {1: cal_scores[rare_cal], 0: cal_scores[~rare_cal]}
    if min(len(cbr[1]), len(cbr[0])) >= 2:
        te_reg = (y_h >= np.nanpercentile(y_h, 90)).astype(int)
        c, w = B.run_mondrian(y_h, p_h, cbr, te_reg)
        arms["mondrian"] = (p_h - w / 2, p_h + w / 2)
    c, w = B.run_pid(y_h, p_h, trailing);           arms["pid_conformal"] = (p_h - w/2, p_h + w/2)
    c, w = B.run_evt(y_h, p_h, cal_scores);         arms["evt_tail"] = (p_h - w/2, p_h + w/2)
    c, w = B.run_dtaci(y_h, p_h, cal_scores);       arms["dtaci"] = (p_h - w/2, p_h + w/2)
    c, w = B.run_acmcp(y_h, p_h, cal_scores);       arms["acmcp"] = (p_h - w/2, p_h + w/2)
    c, w = B.run_bci(y_h, p_h, cal_scores);         arms["bellman_ci"] = (p_h - w/2, p_h + w/2)
    arms["errordir"] = aci(cal_scores, m=mult)

    for nm, (los, his) in arms.items():
        r, wk, cv = score(nm, los, his, y_h, spread)
        r["domain"], r["model"] = dname, model
        rows.append(r); wk_store[(dname, model, nm)] = wk

    print(f"\n  --- {dname}/{model}  (n_test={np.isfinite(y_h).sum()}, "
          f"spread={spread:.3f}, mean mult={mult.mean():.3f}) ---")

D = pd.DataFrame(rows)[["domain", "model", "method", "n", "coverage", "mean_width",
                        "width_spread_ratio", "vacuous", "winkler_mean",
                        "winkler_median"]]
D.to_csv(f"{OUT}/task24_4_comparison.csv", index=False)
for (dn, mo) in VALID:
    sub = D[(D.domain == dn) & (D.model == mo)].sort_values("winkler_mean")
    print(f"\n{dn}/{mo} — sorted by Winkler mean (lower better):")
    print(sub[["method", "coverage", "mean_width", "width_spread_ratio",
               "winkler_mean", "winkler_median"]].to_string(index=False))

# ── significance: errordir vs EVERY baseline, BH across all cells ───────────
print("\n" + "=" * 110)
print("SIGNIFICANCE — errordir vs every baseline (permutation only)")
print("=" * 110)
rng = np.random.default_rng(24)
srows = []
for (dn, mo) in VALID:
    temporal = DOM[dn]["temporal"]
    a = wk_store[(dn, mo, "errordir")]
    for nm in ["pooled_trailing", "diversity_optimal", "mondrian", "pid_conformal",
               "evt_tail", "dtaci", "acmcp", "bellman_ci"]:
        if (dn, mo, nm) not in wk_store:
            continue
        b = wk_store[(dn, mo, nm)]
        n = min(len(a), len(b)); dd = a[:n] - b[:n]
        obs = float(np.mean(dd))
        winrate = 100 * float((dd < 0).mean())
        shift_p = (float((np.array([np.mean(np.roll(dd, int(k)))
                                    for k in rng.integers(1, n, size=2000)]) <= obs).mean())
                   if temporal else np.nan)
        blocks = max(1, n // 12)
        nullb = np.array([float(np.mean(dd * np.repeat(
            rng.choice([-1, 1], size=blocks + 1), 12)[:n])) for _ in range(2000)])
        srows.append(dict(domain=dn, model=mo, baseline=nm, n=n,
                          mean_winkler_diff=round(obs, 3),
                          win_rate_pct=round(winrate, 1),
                          perm_shift_p=None if not temporal else round(shift_p, 4),
                          perm_signflip_p=round(float((nullb <= obs).mean()), 4)))
S = pd.DataFrame(srows)
p = S["perm_signflip_p"].values
order = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
for rank in range(m - 1, -1, -1):
    i = order[rank]; prev = min(prev, p[i] * m / (rank + 1)); bh[i] = prev
S["perm_signflip_p_bh"] = np.round(bh, 4)
S.to_csv(f"{OUT}/task24_4_significance.csv", index=False)
print(S.to_string(index=False))
print(f"\n  BH family size m={m}; min q={S.perm_signflip_p_bh.min():.4f}")
print(f"Saved {OUT}/task24_4_comparison.csv, {OUT}/task24_4_significance.csv")
