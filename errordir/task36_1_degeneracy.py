"""
TASK 36, Item 1 — is delta_i actually constant across test points for the four
ridge fits Task 34 reported?

DIRECT MEASUREMENT on the exact four fits (insurance/ridge/beta,
insurance/ridge/kNN, energy/ridge/beta, energy/ridge/kNN), reproducing Task 34's
construction byte-for-byte, plus a tree fit as a positive control (a tree's
delta_i SHOULD vary, so a near-zero CV there would mean the measurement itself
is broken).

The quantity:
    delta_i(v) = | f(x_i + h*v) - f(x_i - h*v) |
reported as coefficient of variation  CV = sd(delta) / mean(delta).

For a composition of affine maps (which is what the ridge path is, per the
pipeline trace) delta_i is IDENTICAL for every i, so CV = 0 exactly up to
floating point. Any post-hoc nonlinearity would make CV > 0.

Outputs: errordir/task36_1_degeneracy.csv
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
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

OUT = "errordir"
FIT_FRAC, SEED = 0.60, 5
MAG, N_POINTS, N_TEST, POOL_CAP = 0.5, 400, 400, 4000

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


rows = []
for dname in DATA:
    Xp, yp, Xt, yt = load(dname)
    n_pool = len(yp); n_fit = int(round(FIT_FRAC * n_pool))
    FIT_I = np.arange(0, n_fit)

    for kind in ["ridge", "lightgbm"]:          # lightgbm = positive control
        sc0 = StandardScaler().fit(Xp)
        A = sc0.transform(Xp) if kind == "ridge" else Xp
        mk = (lambda: Ridge(alpha=1.0)) if kind == "ridge" else \
             (lambda: lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06,
                                        max_depth=4, verbose=-1, random_state=SEED))
        oof = np.full(n_pool, np.nan)
        for tr, te in rolling_origin_folds(n_pool, n_folds=5):
            m = mk(); m.fit(A[tr], yp[tr]); oof[te] = m.predict(A[te])
        mfull = mk(); mfull.fit(A, yp)
        scores = np.abs(oof - yp); ok = np.isfinite(scores)
        fit_ok = FIT_I[ok[FIT_I]]

        scb = StandardScaler().fit(Xp[fit_ok])
        Zf, ef = scb.transform(Xp[fit_ok]), scores[fit_ok]
        Zt = scb.transform(Xt)
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(Zt), size=min(N_POINTS, len(Zt)), replace=False)
        X0 = Zt[sel]

        def pred(Zs):                            # identical to Task 34's
            raw = scb.inverse_transform(Zs)
            return mfull.predict(sc0.transform(raw) if kind == "ridge" else raw)

        def induced(V, h=MAG):
            V = np.atleast_2d(V)
            if V.shape[0] == 1:
                V = np.repeat(V, len(X0), axis=0)
            return np.abs(pred(X0 + h * V) - pred(X0 - h * V))

        cands = {}
        b = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Zf, ef).coef_
        cands["beta"] = b / (np.linalg.norm(b) + 1e-12)
        K = int(max(20, np.ceil(len(fit_ok) / 50)))
        nn = NearestNeighbors(n_neighbors=K).fit(Zf)
        dA = ef[nn.kneighbors(Zf)[1]].mean(axis=1)
        gA = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Zf, dA).coef_
        cands["A_local_knn"] = gA / (np.linalg.norm(gA) + 1e-12)

        for cname, V in cands.items():
            dl = induced(V)
            mean_d = float(np.mean(dl)); sd_d = float(np.std(dl))
            cv = sd_d / max(mean_d, 1e-300)
            n_unique = int(len(np.unique(np.round(dl, 12))))
            rel_range = float((dl.max() - dl.min()) / max(mean_d, 1e-300))
            print(f"  {dname:10s} {kind:9s} {cname:12s} "
                  f"mean={mean_d:.6g}  sd={sd_d:.3e}  CV={cv:.3e}  "
                  f"unique={n_unique}/{len(dl)}  rel_range={rel_range:.3e}")
            rows.append(dict(domain=dname, model=kind,
                             model_class=("linear" if kind == "ridge" else "tree"),
                             mechanism=cname, n_points=len(dl),
                             delta_mean=mean_d, delta_sd=sd_d,
                             delta_CV=cv, n_unique_deltas=n_unique,
                             rel_range=rel_range,
                             degenerate=bool(cv < 1e-10)))

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task36_1_degeneracy.csv", index=False)
print("\n" + "=" * 100)
print("ITEM 1 — is delta_i constant? (CV ~ 0 => DS cannot discriminate)")
print("=" * 100)
print(D[["domain", "model", "model_class", "mechanism", "delta_mean",
         "delta_CV", "n_unique_deltas", "degenerate"]].to_string(index=False))
print("\nBy model class:")
print(D.groupby("model_class")[["delta_CV", "n_unique_deltas"]].mean().to_string())
print(f"\nDEGENERATE (CV<1e-10): {int(D.degenerate.sum())} of {len(D)}")
print(f"  linear: {int(D[D.model_class=='linear'].degenerate.sum())}"
      f"/{len(D[D.model_class=='linear'])}   "
      f"tree: {int(D[D.model_class=='tree'].degenerate.sum())}"
      f"/{len(D[D.model_class=='tree'])}")
print(f"\nSaved {OUT}/task36_1_degeneracy.csv")
