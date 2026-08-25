"""
TASK 33, Item 1 — measure tree-forecaster perturbation responsiveness DIRECTLY.

NO difficulty function is involved in this item. No beta, no kNN, no
disagreement. This measures a property of the forecasters themselves: when you
apply the exact perturbations Task 21's Check 2 uses, does the model's own
output change at all?

METHOD — the perturbation is identical to Check 2's:
  * standardized feature space (fit on FIT split)
  * magnitude MAG = 0.5 SD, applied as x +/- MAG*v for a unit direction v
  * 200 random unit directions per point (the same null Check 2 builds)
  * 40 sampled test points per fit (Check 2's sample size)
For each (point, direction) pair, record whether
      |pred(x + MAG*v) - pred(x - MAG*v)| == 0    (to float tolerance)
i.e. whether the two-sided perturbation left the forecaster's output unchanged.

Reported per fit:
  zero_rate          fraction of (point, direction) pairs with EXACTLY no change
  near_zero_rate     fraction with change < 1e-6 * (target SD)  -- a tolerance
                     band, since a tree can change by a numerically tiny amount
                     if a single deep split flips
  median_rel_change  median |delta| relative to the target SD, a scale-free
                     measure of how much the output actually moves
  frac_points_dead   fraction of TEST POINTS for which >=95% of directions
                     produce zero change (a point sitting in a wide leaf)

RIDGE CONTROL on the same domains: ridge is linear and continuous, so its
zero_rate should be ~0 by construction. It is the calibration for reading the
tree numbers.

Domains/models are exactly those from Tasks 24-25 where beta failed 0/6:
  insurance x {xgboost, lightgbm, ridge}
  energy    x {xgboost, lightgbm, ridge}
plus climate x {gradboost, ridge} from Task 24.

Outputs: errordir/task33_1_responsiveness.csv
"""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg")); sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs; ensemble_stubs.install()
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

OUT = "errordir"
FIT_FRAC, SEED = 0.60, 5
N_DIRS, MAG, N_POINTS = 200, 0.5, 40
N_TEST, POOL_CAP = 400, 4000

_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    import domain_climate as CL

DATA = {"insurance": "data/domains/task25/insurance.csv",
        "energy": "data/domains/task25/energy.csv"}


def load(dname):
    if dname == "climate":
        return CL.X_pool, CL.y_pool, CL.X_test, np.asarray(CL.Y_TEST, float)
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
    if kind == "lightgbm":
        return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06,
                                 max_depth=4, verbose=-1, random_state=SEED)
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                         max_depth=4, random_state=23)


PLAN = [("insurance", ["xgboost", "lightgbm", "ridge"]),
        ("energy", ["xgboost", "lightgbm", "ridge"]),
        ("climate", ["gradboost", "ridge"])]

rows = []
for dname, kinds in PLAN:
    Xp, yp, Xt, yt = load(dname)
    n_pool = len(yp); n_fit = int(round(FIT_FRAC * n_pool))
    fit_i = np.arange(0, n_fit)
    fit_i = fit_i[np.isfinite(yp[fit_i])]
    tgt_sd = float(np.std(yt[np.isfinite(yt)]))

    for kind in kinds:
        sc0 = StandardScaler().fit(Xp)
        A = sc0.transform(Xp) if kind == "ridge" else Xp
        ok = np.isfinite(yp)
        m = make_model(kind); m.fit(A[ok], yp[ok])

        # perturbation happens in the SAME standardized space Check 2 uses
        scb = StandardScaler().fit(Xp[fit_i])
        Xt_s = scb.transform(Xt)
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(Xt_s), size=min(N_POINTS, len(Xt_s)), replace=False)
        X0 = Xt_s[sel]

        dirs = rng.normal(size=(N_DIRS, X0.shape[1]))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        deltas = np.empty((len(sel), N_DIRS))
        for j, v in enumerate(dirs):
            rp = scb.inverse_transform(X0 + MAG * v)
            rm = scb.inverse_transform(X0 - MAG * v)
            Qp = sc0.transform(rp) if kind == "ridge" else rp
            Qm = sc0.transform(rm) if kind == "ridge" else rm
            deltas[:, j] = np.abs(m.predict(Qp) - m.predict(Qm))

        zero_rate = float(np.mean(deltas == 0.0))
        near_zero = float(np.mean(deltas < 1e-6 * max(tgt_sd, 1e-12)))
        med_rel = float(np.median(deltas) / max(tgt_sd, 1e-12))
        per_point_zero = (deltas == 0.0).mean(axis=1)
        frac_dead = float(np.mean(per_point_zero >= 0.95))

        print(f"  {dname:10s} {kind:9s} zero={zero_rate:.4f}  "
              f"near_zero={near_zero:.4f}  med_rel_change={med_rel:.5f}  "
              f"dead_points={frac_dead:.3f}")
        rows.append(dict(domain=dname, model=kind,
                         model_class=("linear" if kind == "ridge" else "tree"),
                         n_points=len(sel), n_dirs=N_DIRS,
                         zero_rate=round(zero_rate, 5),
                         near_zero_rate=round(near_zero, 5),
                         median_rel_change=round(med_rel, 6),
                         frac_points_dead=round(frac_dead, 4),
                         target_sd=round(tgt_sd, 4)))

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/task33_1_responsiveness.csv", index=False)
print("\n" + "=" * 100)
print("ITEM 1 — perturbation responsiveness of the FORECASTER ITSELF")
print("=" * 100)
print(R[["domain", "model", "model_class", "zero_rate", "near_zero_rate",
         "median_rel_change", "frac_points_dead"]].to_string(index=False))
print("\nBy model class:")
print(R.groupby("model_class")[["zero_rate", "near_zero_rate",
                                "median_rel_change", "frac_points_dead"]]
      .mean().round(5).to_string())
print(f"\nSaved {OUT}/task33_1_responsiveness.csv")
