"""
TASK 35, Item 2 — locality gate.

Fits D(x) with the Silverman bandwidth fixed in Item 1, then checks whether
grad D(x) GENUINELY VARIES across points before any causal test is allowed.

CRITERION (fixed in Item 1, not adjusted here):
    mean pairwise cosine similarity between per-point unit gradients
      LOCAL     if < 0.90
      NOT LOCAL if >= 0.90
    secondary (reported, not gating): fraction of points whose gradient is
      >30 degrees from the mean gradient direction.

Task 33's median-heuristic bandwidth is computed alongside for comparison, so
the effect of the fix is visible.

GUARDRAIL: if ridge fails this check, tree domains must NOT proceed to Item 3.

Outputs: errordir/task35_2_locality_check.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg")); sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs; ensemble_stubs.install()
from domain_common import rolling_origin_folds
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

OUT = "errordir"
FIT_FRAC, SEED = 0.60, 5
N_POINTS, N_TEST, POOL_CAP, SUB = 400, 400, 4000, 1500
COS_BAR = 0.90

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
    """Dimension-adjusted Silverman, per Item 1. FIT data only."""
    n, d = Z.shape
    sigma = float(np.mean(np.std(Z, axis=0)))
    return sigma * (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0)) * n ** (-1.0 / (d + 4.0))


def median_h(Z, rng):
    idx = rng.choice(len(Z), size=min(600, len(Z)), replace=False)
    S = Z[idx]
    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(np.median(d2[np.triu_indices_from(d2, k=1)])))


def nw_grad(Xq, Zf, ef, h):
    """Nadaraya-Watson D(x) and closed-form gradient (same form as Task 33)."""
    diff = Zf[None, :, :] - Xq[:, None, :]
    d2 = (diff ** 2).sum(-1)
    K = np.exp(-d2 / (h ** 2)); S = K.sum(1) + 1e-300
    D = (K * ef[None, :]).sum(1) / S
    wm_e = ((K * ef[None, :])[:, :, None] * diff).sum(1) / S[:, None]
    wm = (K[:, :, None] * diff).sum(1) / S[:, None]
    return D, (2.0 / h ** 2) * (wm_e - D[:, None] * wm), K, S


def locality_stats(G):
    """Mean pairwise cosine similarity + fraction >30deg from mean direction."""
    n = np.linalg.norm(G, axis=1, keepdims=True)
    live = (n.ravel() > 1e-12)
    if live.sum() < 2:
        return 1.0, 0.0, float(live.mean())
    U = G[live] / n[live]
    C = U @ U.T
    iu = np.triu_indices_from(C, k=1)
    mean_cos = float(np.mean(np.abs(C[iu])))
    gm = U.mean(0); gm /= (np.linalg.norm(gm) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.abs(U @ gm), 0, 1)))
    return mean_cos, float((ang > 30).mean()), float(live.mean())


rows = []
for dname in DATA:
    Xp, yp, Xt, yt = load(dname)
    n_pool = len(yp); n_fit = int(round(FIT_FRAC * n_pool))
    FIT_I = np.arange(0, n_fit)

    for kind in ["ridge", "xgboost", "lightgbm"]:
        sc0 = StandardScaler().fit(Xp)
        A = sc0.transform(Xp) if kind == "ridge" else Xp
        oof = np.full(n_pool, np.nan)
        for tr, te in rolling_origin_folds(n_pool, n_folds=5):
            m = make_model(kind); m.fit(A[tr], yp[tr]); oof[te] = m.predict(A[te])
        scores = np.abs(oof - yp); ok = np.isfinite(scores)
        fit_ok = FIT_I[ok[FIT_I]]

        scb = StandardScaler().fit(Xp[fit_ok])
        Zf_all, ef_all = scb.transform(Xp[fit_ok]), scores[fit_ok]
        rng = np.random.default_rng(SEED)
        if len(Zf_all) > SUB:
            s = rng.choice(len(Zf_all), size=SUB, replace=False)
            Zf, ef = Zf_all[s], ef_all[s]
        else:
            Zf, ef = Zf_all, ef_all

        h_sil = silverman_h(Zf)
        h_med = median_h(Zf, rng)

        Zt = scb.transform(Xt)
        sel = rng.choice(len(Zt), size=min(N_POINTS, len(Zt)), replace=False)
        X0 = Zt[sel]

        for hname, h in [("silverman", h_sil), ("median_task33", h_med)]:
            D, G, K, S = nw_grad(X0, Zf, ef, h)
            mean_cos, frac30, frac_live = locality_stats(G)
            # effective neighbours: how many FIT points meaningfully contribute
            eff_n = float(np.mean((K.sum(1) ** 2) / ((K ** 2).sum(1) + 1e-300)))
            is_local = bool(mean_cos < COS_BAR)
            print(f"  {dname:10s} {kind:9s} {hname:14s} h={h:7.4f}  "
                  f"eff_nbrs={eff_n:8.1f}/{len(Zf)}  mean|cos|={mean_cos:.4f}  "
                  f"frac>30deg={frac30:.3f}  -> {'LOCAL' if is_local else 'NOT LOCAL'}")
            rows.append(dict(domain=dname, model=kind,
                             model_class=("linear" if kind == "ridge" else "tree"),
                             bandwidth_rule=hname, h=round(h, 5),
                             n_fit_kernel=len(Zf),
                             eff_neighbours=round(eff_n, 2),
                             eff_nbr_frac=round(eff_n / len(Zf), 4),
                             mean_abs_cosine=round(mean_cos, 4),
                             frac_grad_gt30deg=round(frac30, 4),
                             frac_live_grad=round(frac_live, 4),
                             is_local=is_local))

L = pd.DataFrame(rows)
L.to_csv(f"{OUT}/task35_2_locality_check.csv", index=False)
print("\n" + "=" * 104)
print("ITEM 2 — locality gate (criterion: mean |cosine| < 0.90)")
print("=" * 104)
print(L[["domain", "model", "model_class", "bandwidth_rule", "h",
         "eff_neighbours", "mean_abs_cosine", "frac_grad_gt30deg",
         "is_local"]].to_string(index=False))

sil = L[L.bandwidth_rule == "silverman"]
print("\nSILVERMAN only — the rule fixed in Item 1:")
print(sil.groupby("model_class")[["h", "eff_nbr_frac", "mean_abs_cosine",
                                  "is_local"]].mean().to_string())
ridge_ok = bool(sil[sil.model_class == "linear"].is_local.all())
print(f"\n  RIDGE CONTROL LOCAL: {ridge_ok}")
print(f"  -> Item 3 {'MAY proceed' if ridge_ok else 'MUST NOT proceed (guardrail)'}")
print(f"\nSaved {OUT}/task35_2_locality_check.csv")
