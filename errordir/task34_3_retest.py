"""
TASK 34, Item 3 — apply the VERIFIED DS test to every tree fit and ridge control
from Tasks 32-33, alongside the original Check 2, in one table.

Mechanisms retested (all from Tasks 32-33):
  beta            global linear projection      (Tasks 24/25/32 baseline)
  A_local_knn     local kNN difficulty          (Task 32 Candidate A)
  B_disagreement  ensemble disagreement direct  (Task 32 Candidate B)
  D_local_nw      locally-varying Nadaraya-Watson, per-point gradient (Task 33)

Domains x models: insurance/energy x {xgboost, lightgbm, ridge}.
All construction (splits, K, bandwidth, disagreement members) is copied
unchanged from Tasks 32-33 -- nothing is re-tuned here. Only the TEST changes.

Outputs: errordir/task34_3_retest.csv
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
import xgboost as xgb
import lightgbm as lgb

OUT = "errordir"
FIT_FRAC, SEED = 0.60, 5
N_DIRS, MAG, N_POINTS = 200, 0.5, 400
N_TEST, POOL_CAP, N_BOOT, SUB = 400, 4000, 20, 1500
S_LO, S_HI = 0.1, 10.0

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


def members(kind, model, Xq, Xf, yf):
    if kind in ("xgboost", "lightgbm"):
        P = []
        for k in range(20, 201, 20):
            P.append(model.predict(Xq, iteration_range=(0, k)) if kind == "xgboost"
                     else model.predict(Xq, num_iteration=k))
        return np.stack(P, axis=1)
    rng = np.random.default_rng(SEED); P = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(yf), size=len(yf), replace=True)
        P.append(Ridge(alpha=1.0).fit(Xf[idx], yf[idx]).predict(Xq))
    return np.stack(P, axis=1)


def nw_and_grad(Xq, Zf, ef, h):
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
    FIT_I, CAL_I = np.arange(0, n_fit), np.arange(n_fit, n_pool)

    for kind in ["xgboost", "lightgbm", "ridge"]:
        sc0 = StandardScaler().fit(Xp)
        A = sc0.transform(Xp) if kind == "ridge" else Xp
        oof = np.full(n_pool, np.nan)
        for tr, te in rolling_origin_folds(n_pool, n_folds=5):
            m = make_model(kind); m.fit(A[tr], yp[tr]); oof[te] = m.predict(A[te])
        mfull = make_model(kind); mfull.fit(A, yp)
        scores = np.abs(oof - yp); ok = np.isfinite(scores)
        fit_ok = FIT_I[ok[FIT_I]]

        scb = StandardScaler().fit(Xp[fit_ok])
        Zf = scb.transform(Xp[fit_ok]); ef = scores[fit_ok]
        Zt = scb.transform(Xt)
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(Zt), size=min(N_POINTS, len(Zt)), replace=False)
        X0 = Zt[sel]
        e_test = np.abs(mfull.predict(sc0.transform(Xt) if kind == "ridge" else Xt) - yt)[sel]

        def pred(Zs):
            raw = scb.inverse_transform(Zs)
            return mfull.predict(sc0.transform(raw) if kind == "ridge" else raw)

        def induced(V, h=MAG):
            V = np.atleast_2d(V)
            if V.shape[0] == 1:
                V = np.repeat(V, len(X0), axis=0)
            return np.abs(pred(X0 + h * V) - pred(X0 - h * V))

        # ---- candidate direction fields, construction copied from Tasks 32-33
        cands = {}
        b = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Zf, ef).coef_
        cands["beta"] = b / (np.linalg.norm(b) + 1e-12)

        K = int(max(20, np.ceil(len(fit_ok) / 50)))
        nn = NearestNeighbors(n_neighbors=K).fit(Zf)
        dA = ef[nn.kneighbors(Zf)[1]].mean(axis=1)
        gA = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Zf, dA).coef_
        cands["A_local_knn"] = gA / (np.linalg.norm(gA) + 1e-12)

        Pf = members(kind, mfull, A[fit_ok] if kind != "ridge" else sc0.transform(Xp[fit_ok]), A, yp)
        gB = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Zf, Pf.std(axis=1)).coef_
        cands["B_disagreement"] = gB / (np.linalg.norm(gB) + 1e-12)

        rs = np.random.default_rng(SEED)
        idx = rs.choice(len(Zf), size=min(SUB, len(Zf)), replace=False)
        Zs, es = Zf[idx], ef[idx]
        sm = rs.choice(len(Zs), size=min(600, len(Zs)), replace=False)
        d2 = ((Zs[sm][:, None, :] - Zs[sm][None, :, :]) ** 2).sum(-1)
        h = float(np.sqrt(np.median(d2[np.triu_indices_from(d2, k=1)])))
        _, G = nw_and_grad(X0, Zs, es, h)
        cands["D_local_nw"] = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)

        dirs = rng.normal(size=(N_DIRS, X0.shape[1]))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        raw_null = np.array([float(np.mean(induced(u))) for u in dirs])

        for cname, V in cands.items():
            dv = induced(V)
            target = float(np.mean(dv))
            ds_v = stats.spearmanr(dv, e_test).correlation
            ds_null, nmatch = [], 0
            for u, mn in zip(dirs, raw_null):
                if mn <= 0: continue
                s = target / mn
                if not (S_LO <= s <= S_HI): continue
                r = stats.spearmanr(induced(u, h=MAG * s), e_test).correlation
                if np.isfinite(r): ds_null.append(r); nmatch += 1
            ds_null = np.array(ds_null)
            ds_pct = float((ds_null < ds_v).mean()) if len(ds_null) else np.nan
            c2_pct = float((raw_null < target).mean())
            rows.append(dict(domain=dname, model=kind,
                             model_class=("linear" if kind == "ridge" else "tree"),
                             mechanism=cname,
                             check2_percentile=round(c2_pct, 4),
                             check2_pass=bool(c2_pct >= 0.95),
                             DS=None if not np.isfinite(ds_v) else round(float(ds_v), 4),
                             DS_percentile=None if not np.isfinite(ds_pct) else round(ds_pct, 4),
                             n_matched_dirs=nmatch,
                             DS_pass=bool(np.isfinite(ds_pct) and ds_pct >= 0.95)))
            print(f"  {dname:10s} {kind:9s} {cname:15s} "
                  f"C2={c2_pct:.3f}{'P' if c2_pct>=.95 else ' '}  "
                  f"DS={ds_v:+.4f} pct={ds_pct:.3f}{'P' if ds_pct>=.95 else ' '} "
                  f"(matched {nmatch})")

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/task34_3_retest.csv", index=False)
print("\n" + "=" * 104)
print("ITEM 3 — full table: old Check 2 vs new DS test")
print("=" * 104)
print(R.to_string(index=False))
print("\nPass counts by model class:")
print(R.groupby("model_class")[["check2_pass", "DS_pass"]].sum().to_string())
print("\nPass counts by mechanism x class:")
print(R.groupby(["mechanism", "model_class"])[["check2_pass", "DS_pass"]].sum().to_string())
print(f"\nSaved {OUT}/task34_3_retest.csv")
