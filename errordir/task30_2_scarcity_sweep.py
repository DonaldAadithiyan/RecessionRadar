"""
TASK 30, Item 2 — scarcity sweep extended to the CORRECTED RLCP.

Adds Task 29's published RLCP (randomization + (n+1) normalisation + the +inf
atom) to Task 28's calibration-size sweep. Task 28's sweep never included any
localized method; Task 28's own "RLCP" was baseLCP, so this is the first time
errordir is compared against real RLCP under scarcity.

UNBOUNDED-INTERVAL HANDLING — PRE-REGISTERED, decided before running:
  * Winkler MEDIAN reported alongside the mean at every size.
  * An explicit unbounded_rate column for every method at every size, as a
    first-class result -- a rate rising as calibration shrinks is itself the
    finding about graceful degradation.
  * If a MAJORITY of intervals are unbounded at some size, that cell is marked
    uninformative and its median is suppressed rather than computed over
    mostly-undefined values.

DEGRADATION RATE is computed for every method (Task 28's own self-correction:
"wins at n=50" looked like data efficiency until degradation rate showed it was
a fixed head start). Any RLCP disadvantage is checked this way before being read
as "RLCP needs more data".

Outputs: errordir/task30_2_scarcity_sweep.csv
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
import baselines_lib as B
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from task27_1_cqr import cqr_intervals
from task29_1_rlcp import rlcp_intervals, median_heuristic_gamma

OUT = "errordir"
LO, HI = 0.75, 1.25
FIT_FRAC, SEED = 0.60, 5
N_REP = 20
SIZES = [50, 100, 200, 400, None]

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


rows = []
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

    scb = StandardScaler().fit(Xp[fit_ok])
    bm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scb.transform(Xp[fit_ok]),
                                                    scores[fit_ok]).coef_
    bu = bm / (np.linalg.norm(bm) + 1e-12)
    z_fit = scb.transform(Xp[fit_ok]) @ bu
    z_test = scb.transform(Xt) @ bu
    gamma = median_heuristic_gamma(z_fit)      # FIT-only, fixed

    def aci(cs, mult=None):
        a = ALPHA_TARGET; los, his = [], []
        for t in range(len(yt)):
            q = float(np.quantile(cs, np.clip(1 - a, 0, 1)))
            if mult is not None: q *= float(mult[t])
            los.append(pred_test[t] - q); his.append(pred_test[t] + q)
            miss = 1 if (yt[t] < los[-1] or yt[t] > his[-1]) else 0
            a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss), .01, .99))
        return np.array(los), np.array(his)

    def stats_of(los, his):
        fin = np.isfinite(his - los)
        unb = float(1.0 - fin.mean())
        wk = np.array([winkler(yt[t], los[t], his[t]) for t in range(len(yt))])
        cov = 100 * float(np.mean([(los[t] <= yt[t] <= his[t]) for t in range(len(yt))]))
        wmean = float(np.mean(wk)) if np.all(np.isfinite(wk)) else np.nan
        wmed = float(np.median(wk)) if unb < 0.5 else np.nan     # pre-registered
        wid = float(np.mean((his - los)[fin])) if fin.any() else np.nan
        return cov, wid, wmean, wmed, unb

    for size in SIZES:
        reps = 1 if size is None else N_REP
        acc = {m: [] for m in ["errordir", "rlcp", "dtaci", "mondrian", "cqr"]}
        for r in range(reps):
            rng = np.random.default_rng(1000 * r + SEED)
            if size is None:
                sub = cal_ok
            elif size > len(cal_ok):
                continue
            else:
                sub = np.sort(rng.choice(cal_ok, size=size, replace=False))
            cs = scores[sub]
            zc = scb.transform(Xp[sub]) @ bu

            u = np.empty(len(z_test))
            for t in range(len(z_test)):
                ref = np.sort(np.concatenate([zc, z_test[:t]]))
                u[t] = np.searchsorted(ref, z_test[t], side="right") / max(len(ref), 1)
            acc["errordir"].append(stats_of(*aci(cs, mult=LO + (HI - LO) * np.clip(u, 0, 1))))

            lo_r, hi_r, _ = rlcp_intervals(zc, cs, z_test, pred_test, gamma, seed=29 + r)
            acc["rlcp"].append(stats_of(lo_r, hi_r))

            c, w = B.run_dtaci(yt, pred_test, cs)
            acc["dtaci"].append(stats_of(pred_test - w / 2, pred_test + w / 2))

            rare = cs >= np.percentile(cs, 90)
            cbr = {True: cs[rare], False: cs[~rare]}
            if min(len(cbr[True]), len(cbr[False])) >= 2:
                treg = [bool(x) for x in (yt >= np.percentile(yt, 90))]
                c, w = B.run_mondrian(yt, pred_test, cbr, treg)
                acc["mondrian"].append(stats_of(pred_test - w / 2, pred_test + w / 2))

            lo_c, hi_c = cqr_intervals(Xp[fit_ok], yp[fit_ok], Xp[sub], yp[sub], Xt)
            acc["cqr"].append(stats_of(lo_c, hi_c))

        for m, vals in acc.items():
            if not vals: continue
            a = np.array(vals, dtype=float)
            rows.append(dict(
                domain=dom, n_cal=("full" if size is None else size),
                n_cal_actual=(len(cal_ok) if size is None else size),
                method=m, reps=len(vals),
                coverage=round(float(np.nanmean(a[:, 0])), 2),
                mean_width=round(float(np.nanmean(a[:, 1])), 3),
                width_spread_ratio=round(float(np.nanmean(a[:, 1])) / spread, 2),
                winkler_mean=(None if np.all(np.isnan(a[:, 2]))
                              else round(float(np.nanmean(a[:, 2])), 3)),
                winkler_median=(None if np.all(np.isnan(a[:, 3]))
                                else round(float(np.nanmean(a[:, 3])), 3)),
                unbounded_rate=round(float(np.nanmean(a[:, 4])), 4),
                uninformative=bool(np.nanmean(a[:, 4]) >= 0.5)))
        print(f"  {dom:8s} n_cal={str(size):5s} done")

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task30_2_scarcity_sweep.csv", index=False)

for dom in D.domain.unique():
    s = D[D.domain == dom]
    print(f"\n{'='*96}\n{dom.upper()}\n{'='*96}")
    for col in ["winkler_mean", "winkler_median", "coverage", "unbounded_rate"]:
        p = s.pivot(index="n_cal", columns="method", values=col)
        p = p.reindex([50, 100, 200, 400, "full"])
        print(f"\n{col}:"); print(p.to_string())

print("\n" + "=" * 96)
print("DEGRADATION RATE (full -> n_cal=50), the Task 28 self-correction check")
print("=" * 96)
for dom in D.domain.unique():
    s = D[D.domain == dom]
    print(f"\n  {dom.upper()}")
    for m in ["errordir", "rlcp", "dtaci", "mondrian", "cqr"]:
        try:
            f = s[(s.n_cal.astype(str) == "full") & (s.method == m)]
            v = s[(s.n_cal.astype(str) == "50") & (s.method == m)]
            for col in ["winkler_mean", "winkler_median"]:
                fv, vv = f[col].iloc[0], v[col].iloc[0]
                if pd.notna(fv) and pd.notna(vv):
                    print(f"    {m:9s} {col:15s} full={fv:9.3f} n50={vv:9.3f} "
                          f"deg={vv-fv:+8.3f} ({100*(vv-fv)/fv:+6.1f}%)")
                    break
        except Exception:
            pass
print(f"\nSaved {OUT}/task30_2_scarcity_sweep.csv")
