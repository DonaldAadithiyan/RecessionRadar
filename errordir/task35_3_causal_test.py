"""
TASK 35, Item 3 — causal-adjacency test on the locality-confirmed D(x).

INSTRUMENT ROUTING (fixed by the task, not chosen here):
  * Check 2  -- every fit (ridge and tree). For trees its known
                hyper-responsive-null limitation (Task 33) is stated with the
                result; a pass or fail is not treated as fully conclusive.
  * HOS      -- ADDITIONALLY for ridge fits only (Task 37 verified it valid on
                linear models). D(x) makes a difficulty-RANKING claim, so HOS is
                the appropriate second check.
  * DS       -- NOT USED anywhere (Task 36: degenerate on linear models; no
                working tree-specific correction exists).

All six fits passed Item 2's locality gate, so all six are tested.
Bandwidth is Silverman, exactly as fixed in Item 1 and used in Item 2.

Outputs: errordir/task35_3_causal_test.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg")); sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs; ensemble_stubs.install()
from domain_common import rolling_origin_folds
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb

OUT = "errordir"
FIT_FRAC, SEED = 0.60, 5
N_DIRS, MAG, N_POINTS = 200, 0.5, 400
N_TEST, POOL_CAP, SUB = 400, 4000, 1500
N_NULL_HOS, HOLDOUT_FRAC = 200, 0.25
ALPHAS = np.logspace(-3, 3, 25)

DATA = {"insurance": "data/domains/task25/insurance.csv",
        "energy": "data/domains/task25/energy.csv"}


def load(dname):
    d = pd.read_csv(DATA[dname])
    y = d["target"].values.astype(float)
    X = d.drop(columns=["target"]).values.astype(float)
    rng = np.random.default_rng(SEED)
    if dname == "energy":
        keep = np.arange(max(0, len(y) - (POOL_CAP + N_TEST)), len(y))
    else:
        keep = np.sort(rng.choice(len(y), size=min(POOL_CAP + N_TEST, len(y)),
                                  replace=False))
    X, y = X[keep], y[keep]
    return X[:-N_TEST], y[:-N_TEST], X[-N_TEST:], y[-N_TEST:]


def make_model(kind):
    if kind == "ridge":
        return Ridge(alpha=1.0)
    if kind == "xgboost":
        return xgb.XGBRegressor(n_estimators=200, learning_rate=0.06, max_depth=4,
                                subsample=0.9, verbosity=0, random_state=SEED)
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06, max_depth=4,
                             verbose=-1, random_state=SEED)


def silverman_h(Z):
    n, d = Z.shape
    sigma = float(np.mean(np.std(Z, axis=0)))
    return sigma * (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0)) * n ** (-1.0 / (d + 4.0))


def nw_grad(Xq, Zf, ef, h):
    diff = Zf[None, :, :] - Xq[:, None, :]
    d2 = (diff ** 2).sum(-1)
    K = np.exp(-d2 / (h ** 2)); S = K.sum(1) + 1e-300
    D = (K * ef[None, :]).sum(1) / S
    wm_e = ((K * ef[None, :])[:, :, None] * diff).sum(1) / S[:, None]
    wm = (K[:, :, None] * diff).sum(1) / S[:, None]
    return D, (2.0 / h ** 2) * (wm_e - D[:, None] * wm)


rows = []
for dname in DATA:
    Xp, yp, Xt, yt = load(dname)
    n_pool = len(yp); n_fit = int(round(FIT_FRAC * n_pool))
    n_hold = int(round(HOLDOUT_FRAC * n_pool))
    FIT_I = np.arange(0, n_fit)
    HOLD_I = np.arange(n_pool - n_hold, n_pool)
    assert len(np.intersect1d(FIT_I, HOLD_I)) == 0

    for kind in ["ridge", "xgboost", "lightgbm"]:
        sc0 = StandardScaler().fit(Xp)
        A = sc0.transform(Xp) if kind == "ridge" else Xp
        oof = np.full(n_pool, np.nan)
        for tr, te in rolling_origin_folds(n_pool, n_folds=5):
            m = make_model(kind); m.fit(A[tr], yp[tr]); oof[te] = m.predict(A[te])
        mfull = make_model(kind); mfull.fit(A, yp)
        scores = np.abs(oof - yp); ok = np.isfinite(scores)
        fit_ok = FIT_I[ok[FIT_I]]; hold_ok = HOLD_I[ok[HOLD_I]]

        scb = StandardScaler().fit(Xp[fit_ok])
        Zf_all, ef_all = scb.transform(Xp[fit_ok]), scores[fit_ok]
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(Zf_all), size=min(SUB, len(Zf_all)), replace=False) \
            if len(Zf_all) > SUB else np.arange(len(Zf_all))
        Zf, ef = Zf_all[idx], ef_all[idx]
        h = silverman_h(Zf)

        Zt = scb.transform(Xt)
        sel = rng.choice(len(Zt), size=min(N_POINTS, len(Zt)), replace=False)
        X0 = Zt[sel]
        _, G = nw_grad(X0, Zf, ef, h)
        Gu = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)

        # CHECK 1 (unchanged) on the mean gradient direction
        gm = Gu.mean(0); gm /= (np.linalg.norm(gm) + 1e-12)
        pca = PCA(n_components=min(10, Zf.shape[1])).fit(Zf)
        ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(
            gm, pca.components_[0] / np.linalg.norm(pca.components_[0])))), 0, 1)))
        check1 = bool(ang > 30.0)

        def pred(Zs):
            raw = scb.inverse_transform(Zs)
            return mfull.predict(sc0.transform(raw) if kind == "ridge" else raw)

        def induced(V, hh=MAG):
            V = np.atleast_2d(V)
            if V.shape[0] == 1:
                V = np.repeat(V, len(X0), axis=0)
            return np.abs(pred(X0 + hh * V) - pred(X0 - hh * V))

        # CHECK 2, per-point candidate vs per-point random null
        dirs = rng.normal(size=(N_DIRS, X0.shape[1]))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        eff_sig = induced(Gu)
        null = np.empty((len(X0), N_DIRS))
        for j, u in enumerate(dirs):
            null[:, j] = induced(u)
        per_pt = (null < eff_sig[:, None]).mean(1)
        c2 = float(per_pt.mean())
        c2_pass = bool(c2 >= 0.95)

        rec = dict(domain=dname, model=kind,
                   model_class=("linear" if kind == "ridge" else "tree"),
                   h=round(h, 5), angle_pc1_deg=round(ang, 2),
                   check1_pass=check1,
                   check2_percentile=round(c2, 4), check2_pass=c2_pass,
                   check2_caveat=("hyper-responsive null (Task 33) unresolved"
                                  if kind != "ridge" else ""),
                   HOS=None, HOS_null_mean=None, HOS_percentile=None,
                   HOS_pass=None)

        # HOS — ridge only, per the fixed instrument routing
        if kind == "ridge":
            Zh, eh = scb.transform(Xp[hold_ok]), scores[hold_ok]
            Dh, _ = nw_grad(Zh, Zf, ef, h)          # D(x) as a ranking score
            s = abs(float(stats.spearmanr(Dh, eh).correlation))
            rr = np.random.default_rng(SEED)
            ns = []
            for _ in range(N_NULL_HOS):
                efp = rr.permutation(ef)
                Dp, _ = nw_grad(Zh, Zf, efp, h)
                v = stats.spearmanr(Dp, eh).correlation
                if np.isfinite(v):
                    ns.append(abs(float(v)))
            ns = np.array(ns)
            pct = float((ns < s).mean()) if len(ns) else np.nan
            rec.update(HOS=round(s, 4), HOS_null_mean=round(float(ns.mean()), 4),
                       HOS_percentile=round(pct, 4), HOS_pass=bool(pct >= 0.95))

        rows.append(rec)
        extra = (f"  HOS={rec['HOS']:.4f} pct={rec['HOS_percentile']:.3f}"
                 if kind == "ridge" else "")
        print(f"  {dname:10s} {kind:9s} angle={ang:5.1f} C2={c2:.3f}"
              f"{'P' if c2_pass else ' '}{extra}")

C = pd.DataFrame(rows)
C.to_csv(f"{OUT}/task35_3_causal_test.csv", index=False)
print("\n" + "=" * 104)
print("ITEM 3 — causal-adjacency results (locality-confirmed fits only)")
print("=" * 104)
print(C[["domain", "model", "model_class", "check1_pass", "check2_percentile",
         "check2_pass", "HOS", "HOS_percentile", "HOS_pass"]].to_string(index=False))
print("\nCheck 2 passes by class:")
print(C.groupby("model_class").check2_pass.agg(["sum", "count"]).to_string())
r = C[C.model_class == "linear"]
print(f"\nHOS (ridge only): {int(r.HOS_pass.fillna(False).sum())}/{len(r)} pass")
print(f"\nSaved {OUT}/task35_3_causal_test.csv")
