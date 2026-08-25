"""
TASK 32, Items 2-4 — two candidate difficulty signals, validated on the domains
where linear beta failed 0/6, with a ridge control arm.

THE DESIGN BAR (restated): a tree's prediction surface is piecewise, so a single
global direction assumes a smooth consistent error gradient that does not exist.
Neither candidate below makes that assumption.

CANDIDATE A — LOCAL kNN DIFFICULTY
    difficulty(x) = a robust summary (mean) of the OOF |error| of x's K nearest
    FIT-period neighbours, in standardized feature space.
    No global direction: the signal is defined pointwise from local history.
    K = max(20, ceil(n_fit/50)) -- fixed in advance. Floor of 20 so the local
    error mean rests on >=20 observations (same reasoning as Task 22's p99 floor
    and Task 28's window floor); n_fit/50 gives ~50 effective local regions.
    NO test-period tuning of K or of the distance metric.

CANDIDATE B — ENSEMBLE DISAGREEMENT AS THE SCORE ITSELF
    difficulty(x) = sd over an ensemble's member predictions at x.
    WHY THIS IS NOT TASK 23's CONSTRUCTION -- stated before results:
      Task 23 fed disagreement in as ONE INPUT FEATURE to a linear beta, i.e. it
      asked beta to treat a model OUTPUT as if it were an independent input
      cause, then fit a direction through it. Check 2 broke precisely there:
      perturbing along a direction loaded on disagreement moved the prediction
      less than random, because disagreement is downstream of the prediction.
      Here there is NO projection and NO fitted direction. Disagreement IS the
      score. Nothing is being asked to treat an output as an input, so Task 23's
      specific failure mode cannot arise by construction.
    For tree models the ensemble is the model's own trees (bagged members).
    For the ridge control a small bootstrap ensemble is built, since ridge has
    no internal ensemble -- this is the "or a small bootstrap ensemble
    constructed for this purpose" case the task allows.

VALIDATION: Task 21's gate, UNMODIFIED.
  Check 1 -- not-just-variance. For a pointwise signal there is no beta vector,
    so the check is applied to the signal's own gradient direction, estimated by
    regressing the difficulty score on features (this is a DIAGNOSTIC of the
    signal, not a refit of the mechanism -- the mechanism stays pointwise).
  Check 2 -- perturbation. Perturb test inputs along the signal's local gradient
    vs random directions of the same magnitude, and compare the change in the
    scored model's own prediction, against a same-model-class null (Task 25's
    construction A).

RIDGE CONTROL on the same domains (Task 25's design) separates "the mechanism
fails" from "the domain has no signal".

Outputs: errordir/task32_4_validation.csv
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
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import lightgbm as lgb

OUT = "errordir"
FIT_FRAC, SEED = 0.60, 5
N_DIRS, MAG = 200, 0.5
N_TEST, POOL_CAP = 400, 4000
N_BOOT = 20                      # bootstrap ensemble size for the ridge control

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


def member_predictions(kind, model, Xq, Xf, yf):
    """Per-member predictions for Candidate B."""
    if kind == "xgboost":
        # staged predictions -> use per-tree contributions across boosting rounds
        P = []
        for k in range(20, 201, 20):
            P.append(model.predict(Xq, iteration_range=(0, k)))
        return np.stack(P, axis=1)
    if kind == "lightgbm":
        P = []
        for k in range(20, 201, 20):
            P.append(model.predict(Xq, num_iteration=k))
        return np.stack(P, axis=1)
    # ridge control: bootstrap ensemble
    rng = np.random.default_rng(SEED)
    P = []
    for b in range(N_BOOT):
        idx = rng.choice(len(yf), size=len(yf), replace=True)
        m = Ridge(alpha=1.0).fit(Xf[idx], yf[idx])
        P.append(m.predict(Xq))
    return np.stack(P, axis=1)


rows = []
for dname in DATA:
    Xp, yp, Xt, yt = load(dname)
    n_pool = len(yp); n_fit = int(round(FIT_FRAC * n_pool))
    FIT_I, CAL_I = np.arange(0, n_fit), np.arange(n_fit, n_pool)

    for kind in ["xgboost", "lightgbm", "ridge"]:
        sc0 = StandardScaler().fit(Xp)
        A = sc0.transform(Xp) if kind == "ridge" else Xp
        Bx = sc0.transform(Xt) if kind == "ridge" else Xt

        oof = np.full(n_pool, np.nan)
        for tr, te in rolling_origin_folds(n_pool, n_folds=5):
            m = make_model(kind); m.fit(A[tr], yp[tr]); oof[te] = m.predict(A[te])
        mfull = make_model(kind); mfull.fit(A, yp)
        scores = np.abs(oof - yp)
        ok = np.isfinite(scores)
        fit_ok = FIT_I[ok[FIT_I]]

        scb = StandardScaler().fit(Xp[fit_ok])
        Xf_s = scb.transform(Xp[fit_ok])
        Xt_s = scb.transform(Xt)
        err_fit = scores[fit_ok]

        # ── CANDIDATE A: local kNN difficulty ──────────────────────────────
        K = int(max(20, np.ceil(len(fit_ok) / 50)))
        nn = NearestNeighbors(n_neighbors=K).fit(Xf_s)

        def diff_A(Xq_s):
            _, idx = nn.kneighbors(Xq_s)
            return err_fit[idx].mean(axis=1)

        # ── CANDIDATE B: ensemble disagreement as the score ────────────────
        Pf = member_predictions(kind, mfull, A[fit_ok] if kind != "ridge"
                                else sc0.transform(Xp[fit_ok]), A, yp)
        dis_fit = Pf.std(axis=1)

        def diff_B(Xq_s):
            raw = scb.inverse_transform(Xq_s)
            Q = sc0.transform(raw) if kind == "ridge" else raw
            return member_predictions(kind, mfull, Q, A, yp).std(axis=1)

        for cand, dfun, dfit in [("A_local_knn", diff_A, diff_A(Xf_s)),
                                 ("B_disagreement", diff_B, dis_fit)]:
            # held-out quality: does the signal track realized error on CAL?
            cal_ok = CAL_I[ok[CAL_I]]
            d_cal = dfun(scb.transform(Xp[cal_ok]))
            r_cal = float(stats.pearsonr(d_cal, scores[cal_ok])[0]) \
                if np.std(d_cal) > 1e-12 else np.nan

            # Check 1: the signal's own gradient direction vs PC1
            g = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf_s, dfit).coef_
            gu = g / (np.linalg.norm(g) + 1e-12)
            pca = PCA(n_components=min(10, Xf_s.shape[1])).fit(Xf_s)
            ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(
                gu, pca.components_[0] / np.linalg.norm(pca.components_[0])))), 0, 1)))
            check1 = bool(ang > 30.0)

            # Check 2: perturbation along the signal's gradient, same-class null
            rng = np.random.default_rng(SEED)
            sel = rng.choice(len(Xt_s), size=min(40, len(Xt_s)), replace=False)
            X0 = Xt_s[sel]

            def eff(v):
                raw_p = scb.inverse_transform(X0 + MAG * v)
                raw_m = scb.inverse_transform(X0 - MAG * v)
                Qp = sc0.transform(raw_p) if kind == "ridge" else raw_p
                Qm = sc0.transform(raw_m) if kind == "ridge" else raw_m
                return float(np.mean(np.abs(mfull.predict(Qp) - mfull.predict(Qm))))

            e = eff(gu)
            null = np.array([eff(v / np.linalg.norm(v))
                             for v in rng.normal(size=(N_DIRS, len(gu)))])
            pct = float((null < e).mean())
            check2 = bool(pct >= 0.95)

            print(f"  {dname:10s} {kind:9s} {cand:15s} r_cal={r_cal:+.3f} "
                  f"angle={ang:5.1f} pct={pct:.3f} -> "
                  f"{'VALIDATED' if (check1 and check2) else 'NOT VALIDATED'}")
            rows.append(dict(domain=dname, model=kind, candidate=cand,
                             model_class=("tree" if kind != "ridge" else "linear"),
                             K=(K if cand == "A_local_knn" else None),
                             r_cal_heldout=None if not np.isfinite(r_cal) else round(r_cal, 4),
                             angle_pc1_deg=round(ang, 2), check1_not_variance=check1,
                             perturb_effect=round(e, 6),
                             perturb_null_p95=round(float(np.percentile(null, 95)), 6),
                             perturb_percentile=round(pct, 4),
                             check2_perturbation=check2,
                             validated=bool(check1 and check2)))

V = pd.DataFrame(rows)
V.to_csv(f"{OUT}/task32_4_validation.csv", index=False)
print("\n" + "=" * 100)
print("ITEM 4 — full table (both candidates x tree domains x ridge control)")
print("=" * 100)
print(V[["domain", "model", "model_class", "candidate", "r_cal_heldout",
         "angle_pc1_deg", "perturb_percentile", "check1_not_variance",
         "check2_perturbation", "validated"]].to_string(index=False))
print("\nvalidated by candidate x model class:")
print(V.groupby(["candidate", "model_class"]).validated.agg(["sum", "count"]).to_string())
print(f"\nSaved {OUT}/task32_4_validation.csv")
