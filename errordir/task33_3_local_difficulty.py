"""
TASK 33, Item 3 — a locally-varying difficulty function D(x), with a per-point
direction grad D(x) for the perturbation check.

MOTIVATION (Item 2): trees respond to essentially EVERY direction (zero-rate
0.00003, 2.7x ridge's median change), which makes Check 2's random-direction
null high and flat. All three mechanisms tried so far supply ONE direction per
fit. Against a flat high null that is the wrong shape of candidate. A
locally-varying D gives a DIFFERENT direction at each test point, which is a
mechanistically distinct proposition.

ESTIMATOR — stated with its parameters fixed in advance from FIT-period
reasoning, before any test-period result is seen:

  D(x) = Nadaraya-Watson kernel regression of |error| on x, in standardized
         FIT-space, with a Gaussian kernel:
             D(x) = sum_i K_h(x, x_i) e_i / sum_i K_h(x, x_i)
         grad D(x) obtained in CLOSED FORM (not finite differences):
             grad D = (2/h^2) * [ sum_i K_i e_i (x_i - x) / sum_i K_i
                                  - D(x) * sum_i K_i (x_i - x) / sum_i K_i ]
         i.e. (2/h^2) * ( weighted-mean-of-(e*offset) - D * weighted-mean-offset ).

  WHY NADARAYA-WATSON rather than a spline or small NN:
    * it has a closed-form gradient, so the direction is exact rather than a
      finite-difference approximation whose step size would be a second free
      parameter to justify;
    * it is genuinely local by construction -- D and grad D at x depend only on
      neighbours weighted by distance, which is the property Item 2 argues is
      needed;
    * it introduces exactly ONE free parameter (h), keeping the fixed-in-advance
      burden minimal. A spline needs knot placement; a NN needs architecture,
      depth, optimiser and stopping -- each a tuning surface this project's
      guardrails would require justifying without test data.

  BANDWIDTH h = median pairwise distance among FIT points (the standard median
  heuristic), computed on the FIT split ONLY. This is the same estimator-free
  default Task 29 used for RLCP's gamma. NOT tuned against test results; no
  sweep over h is run anywhere in this task.

VALIDATION: Task 21's gate, UNMODIFIED.
  Check 1 -- not-just-variance, applied to the MEAN gradient direction (a
    summary diagnostic of D; the mechanism itself stays per-point).
  Check 2 -- per-point: for each sampled test point, perturb along ITS OWN
    grad D(x) and compare against that same point's random-direction null.
    This is the key difference from Tasks 21-32: the null is evaluated
    per-point, and the candidate direction varies per-point, so a direction
    that is right locally is not averaged away.
    The reported percentile is the mean over points of the per-point percentile.

RIDGE CONTROL on the same domains, per the guardrail.

Outputs: errordir/task33_3_local_difficulty.csv
"""
import os, sys, io, contextlib, warnings
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
N_DIRS, MAG, N_POINTS = 200, 0.5, 40
N_TEST, POOL_CAP = 400, 4000
SUB = 1500          # cap on FIT points used in the kernel, for tractability

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


def median_h(Z, rng):
    idx = rng.choice(len(Z), size=min(600, len(Z)), replace=False)
    S = Z[idx]
    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices_from(d2, k=1)
    return float(np.sqrt(np.median(d2[iu])))


def nw_and_grad(Xq, Zf, ef, h):
    """Nadaraya-Watson D(x) and its closed-form gradient. Xq: (m,d)."""
    diff = Zf[None, :, :] - Xq[:, None, :]              # (m, n, d)
    d2 = (diff ** 2).sum(-1)                            # (m, n)
    K = np.exp(-d2 / (h ** 2))
    S = K.sum(1) + 1e-300
    D = (K * ef[None, :]).sum(1) / S
    wm_e = ((K * ef[None, :])[:, :, None] * diff).sum(1) / S[:, None]
    wm = (K[:, :, None] * diff).sum(1) / S[:, None]
    G = (2.0 / h ** 2) * (wm_e - D[:, None] * wm)
    return D, G


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
        fit_ok, cal_ok = FIT_I[ok[FIT_I]], CAL_I[ok[CAL_I]]

        scb = StandardScaler().fit(Xp[fit_ok])
        Zf_all = scb.transform(Xp[fit_ok]); ef_all = scores[fit_ok]
        rng = np.random.default_rng(SEED)
        if len(Zf_all) > SUB:
            s = rng.choice(len(Zf_all), size=SUB, replace=False)
            Zf, ef = Zf_all[s], ef_all[s]
        else:
            Zf, ef = Zf_all, ef_all
        h = median_h(Zf, rng)

        # held-out quality on CAL
        Zc = scb.transform(Xp[cal_ok])
        Dc, _ = nw_and_grad(Zc, Zf, ef, h)
        r_cal = float(stats.pearsonr(Dc, scores[cal_ok])[0]) if np.std(Dc) > 1e-12 else np.nan

        Zt = scb.transform(Xt)
        sel = rng.choice(len(Zt), size=min(N_POINTS, len(Zt)), replace=False)
        X0 = Zt[sel]
        D0, G0 = nw_and_grad(X0, Zf, ef, h)
        gn = np.linalg.norm(G0, axis=1, keepdims=True) + 1e-12
        Gu = G0 / gn

        # CHECK 1 on the mean gradient direction
        gmean = Gu.mean(0); gmean /= (np.linalg.norm(gmean) + 1e-12)
        pca = PCA(n_components=min(10, Zf.shape[1])).fit(Zf)
        ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(
            gmean, pca.components_[0] / np.linalg.norm(pca.components_[0])))), 0, 1)))
        check1 = bool(ang > 30.0)

        def pred_at(Zs):
            raw = scb.inverse_transform(Zs)
            return mfull.predict(sc0.transform(raw) if kind == "ridge" else raw)

        # CHECK 2, PER-POINT: each point's own grad vs its own random null
        dirs = rng.normal(size=(N_DIRS, X0.shape[1]))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        eff_sig = np.abs(pred_at(X0 + MAG * Gu) - pred_at(X0 - MAG * Gu))
        null = np.empty((len(sel), N_DIRS))
        for j, v in enumerate(dirs):
            null[:, j] = np.abs(pred_at(X0 + MAG * v) - pred_at(X0 - MAG * v))
        per_point_pct = (null < eff_sig[:, None]).mean(1)
        pct = float(per_point_pct.mean())
        check2 = bool(pct >= 0.95)
        frac_pts_pass = float((per_point_pct >= 0.95).mean())

        print(f"  {dname:10s} {kind:9s} h={h:6.2f} r_cal={r_cal:+.3f} "
              f"angle={ang:5.1f} pct={pct:.3f} pts>=.95={frac_pts_pass:.2f} -> "
              f"{'VALIDATED' if (check1 and check2) else 'NOT VALIDATED'}")
        rows.append(dict(domain=dname, model=kind,
                         model_class=("linear" if kind == "ridge" else "tree"),
                         bandwidth_h=round(h, 4),
                         r_cal_heldout=None if not np.isfinite(r_cal) else round(r_cal, 4),
                         angle_pc1_deg=round(ang, 2), check1_not_variance=check1,
                         mean_perturb_percentile=round(pct, 4),
                         frac_points_pct_ge_95=round(frac_pts_pass, 4),
                         check2_perturbation=check2,
                         validated=bool(check1 and check2)))

V = pd.DataFrame(rows)
V.to_csv(f"{OUT}/task33_3_local_difficulty.csv", index=False)
print("\n" + "=" * 100)
print("ITEM 3 — locally-varying D(x), per-point gradient direction")
print("=" * 100)
print(V.to_string(index=False))
print("\nvalidated by model class:")
print(V.groupby("model_class").validated.agg(["sum", "count"]).to_string())
print(f"\nSaved {OUT}/task33_3_local_difficulty.csv")
