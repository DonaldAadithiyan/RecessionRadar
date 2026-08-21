"""
TASK 28, Item 2 — data-scarcity sweep.

Tests whether conditioning width on z = beta'x reaches a given frontier position
with LESS calibration data than baselines need for the same position.

Uses the ALREADY-VALIDATED 1D beta from climate/ridge and energy/ridge. No new
fitting of beta. Only the calibration set size changes.

n_cal in {50, 100, 200, 400, full}. At each size, errordir is compared against
DtACI, Mondrian and CQR calibrated at the SAME size on the SAME domain.

OOF DISCIPLINE AT EVERY SIZE (the guardrail): the calibration scores are always
out-of-fold, the beta fit always uses the FIT split only, and CQR's quantile
regressors are always fit on FIT and conformalized on the (subsampled) CAL. A
smaller n_cal subsamples the CAL split; it never borrows FIT or TEST rows.

Repeated over N_REP random subsamples per size (except 'full') so the result is
not an artifact of one lucky draw; mean and sd reported.

Outputs: errordir/task28_2_scarcity_sweep.csv
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

OUT = "errordir"
LO, HI = 0.75, 1.25
FIT_FRAC, SEED = 0.60, 5
N_REP = 20
SIZES = [50, 100, 200, 400, None]      # None = full CAL

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
    proj_test = scb.transform(Xt) @ bu

    def aci(cs, mult=None):
        a = ALPHA_TARGET; los, his = [], []
        for t in range(len(yt)):
            q = float(np.quantile(cs, np.clip(1 - a, 0, 1)))
            if mult is not None: q *= float(mult[t])
            los.append(pred_test[t] - q); his.append(pred_test[t] + q)
            miss = 1 if (yt[t] < los[-1] or yt[t] > his[-1]) else 0
            a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss), .01, .99))
        return np.array(los), np.array(his)

    def summarize(los, his):
        wk = np.array([winkler(yt[t], los[t], his[t]) for t in range(len(yt))])
        cov = 100 * float(np.mean([(los[t] <= yt[t] <= his[t]) for t in range(len(yt))]))
        return cov, float(np.mean(his - los)), float(np.mean(wk))

    for size in SIZES:
        reps = 1 if size is None else N_REP
        acc = {m: [] for m in ["errordir", "dtaci", "mondrian", "cqr"]}
        for r in range(reps):
            rng = np.random.default_rng(1000 * r + SEED)
            if size is None:
                sub = cal_ok
            else:
                if size > len(cal_ok): continue
                sub = np.sort(rng.choice(cal_ok, size=size, replace=False))
            cs = scores[sub]

            pc = scb.transform(Xp[sub]) @ bu
            u = np.empty(len(proj_test))
            for t in range(len(proj_test)):
                ref = np.sort(np.concatenate([pc, proj_test[:t]]))
                u[t] = np.searchsorted(ref, proj_test[t], side="right") / max(len(ref), 1)
            acc["errordir"].append(summarize(*aci(cs, mult=LO + (HI - LO) * np.clip(u, 0, 1))))

            c, w = B.run_dtaci(yt, pred_test, cs)
            acc["dtaci"].append(summarize(pred_test - w / 2, pred_test + w / 2))

            rare = cs >= np.percentile(cs, 90)
            cbr = {True: cs[rare], False: cs[~rare]}
            if min(len(cbr[True]), len(cbr[False])) >= 2:
                treg = [bool(x) for x in (yt >= np.percentile(yt, 90))]
                c, w = B.run_mondrian(yt, pred_test, cbr, treg)
                acc["mondrian"].append(summarize(pred_test - w / 2, pred_test + w / 2))

            lo_c, hi_c = cqr_intervals(Xp[fit_ok], yp[fit_ok], Xp[sub], yp[sub], Xt)
            acc["cqr"].append(summarize(lo_c, hi_c))

        for m, vals in acc.items():
            if not vals: continue
            a = np.array(vals)
            rows.append(dict(domain=dom, n_cal=("full" if size is None else size),
                             n_cal_actual=(len(cal_ok) if size is None else size),
                             method=m, reps=len(vals),
                             coverage=round(float(a[:, 0].mean()), 2),
                             coverage_sd=round(float(a[:, 0].std()), 2),
                             mean_width=round(float(a[:, 1].mean()), 3),
                             width_spread_ratio=round(float(a[:, 1].mean()) / spread, 2),
                             winkler_mean=round(float(a[:, 2].mean()), 3),
                             winkler_sd=round(float(a[:, 2].std()), 3)))
        print(f"  {dom:8s} n_cal={str(size):5s} done")

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task28_2_scarcity_sweep.csv", index=False)
print("\n" + "=" * 96)
for dom in D.domain.unique():
    print(f"\n{dom.upper()} — Winkler mean by calibration size (lower better):")
    piv = D[D.domain == dom].pivot(index="n_cal", columns="method", values="winkler_mean")
    piv = piv.reindex([50, 100, 200, 400, "full"])
    print(piv.to_string())
    print(f"{dom.upper()} — coverage by calibration size:")
    piv2 = D[D.domain == dom].pivot(index="n_cal", columns="method", values="coverage")
    print(piv2.reindex([50, 100, 200, 400, "full"]).to_string())
print(f"\nSaved {OUT}/task28_2_scarcity_sweep.csv")
