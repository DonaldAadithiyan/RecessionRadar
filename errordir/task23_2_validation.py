"""
TASK 23, Item 2 — refit beta with ensemble disagreement, re-run Task 21's full
validation gate FRESH on all four horizons.

The gate is NOT inherited from Task 21. A different feature set is a different
model and earns its own verdict, in both directions: 6M may now fail, and
Current/1M/3M may now pass.

Identical protocol to task21_1_beta.py:
  - three-way split: FIT (291) / CAL (254) / TEST — beta fit on FIT only
  - continuous RidgeCV regression of |error| on features, every FIT month
  - CHECK 1: angle(beta, PC1) > 30 degrees
  - CHECK 2: perturbation effect above the 95th pct of a 200-direction null
The ONLY change is the feature matrix: existing features + 3 disagreement
features for that horizon (std, range, maxpair).

Outputs: errordir/task23_2_validation.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg"))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()
from task23_1_disagreement_check import disagreement  # noqa: E402
from task_oof_and_probit import (  # noqa: E402
    X_train as X_train_df, X_test as X_test_df, y_train as y_pool,
    oof_pred as preds_pool_oof, preds_test, ensemble, LABELS, n_pool,
)

OUT = "errordir"
FIT_FRAC, SEED = 0.60, 5
N_PERTURB_DIRS, PERTURB_MAG = 200, 0.5      # identical to Task 21
rng_global = np.random.default_rng(SEED)

D_train, _ = disagreement(X_train_df)
D_test, _ = disagreement(X_test_df)

n_fit = int(round(FIT_FRAC * n_pool))
FIT_IDX, CAL_IDX = np.arange(0, n_fit), np.arange(n_fit, n_pool)
assert len(np.intersect1d(FIT_IDX, CAL_IDX)) == 0

print("=" * 100)
print("TASK 23 Item 2 — beta refit WITH ensemble disagreement; gate re-run fresh")
print("=" * 100)
print(f"  FIT={len(FIT_IDX)}  CAL={len(CAL_IDX)}  TEST={len(X_test_df)}")
print(f"  base features: {X_train_df.shape[1]}  + 3 disagreement features/horizon\n")

rows = []
for h_idx, h in enumerate(LABELS):
    dcols = [f"disag_std_{h}", f"disag_range_{h}", f"disag_maxpair_{h}"]
    Xtr = np.column_stack([X_train_df.values, D_train[dcols].values])
    Xte = np.column_stack([X_test_df.values, D_test[dcols].values])
    FEATS = list(X_train_df.columns) + dcols

    abs_err = np.abs(preds_pool_oof[:, h_idx] - y_pool[:, h_idx])
    fit_ok = FIT_IDX[np.isfinite(abs_err[FIT_IDX])]
    cal_ok = CAL_IDX[np.isfinite(abs_err[CAL_IDX])]

    scaler = StandardScaler().fit(Xtr[fit_ok])
    Xf = scaler.transform(Xtr[fit_ok])
    reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf, abs_err[fit_ok])
    beta = reg.coef_.astype(float)
    bu = beta / (np.linalg.norm(beta) + 1e-12)

    r_fit = stats.pearsonr(Xf @ beta, abs_err[fit_ok]).statistic
    Xc = scaler.transform(Xtr[cal_ok])
    r_cal = stats.pearsonr(Xc @ beta, abs_err[cal_ok]).statistic
    rho_cal = stats.spearmanr(Xc @ beta, abs_err[cal_ok]).correlation

    # CHECK 1
    pca = PCA(n_components=min(10, Xf.shape[1])).fit(Xf)
    angs = [np.degrees(np.arccos(np.clip(
        abs(float(np.dot(bu, pca.components_[k] / np.linalg.norm(pca.components_[k])))),
        0, 1))) for k in range(3)]
    check1 = bool(angs[0] > 30.0)

    # CHECK 2 — perturbation. Perturb the BASE features along beta's base-feature
    # component (the ensemble consumes base features only); disagreement moves
    # as a consequence, which is the point.
    nb = X_train_df.shape[1]
    Xt_s = scaler.transform(Xte)
    sel = rng_global.choice(len(Xte), size=min(30, len(Xte)), replace=False)

    def pred_at(Xs_mod):
        full = scaler.inverse_transform(Xs_mod)
        Xdf = pd.DataFrame(full[:, :nb], columns=list(X_train_df.columns))
        return ensemble.predict(Xdf)[:, h_idx]

    dp = pred_at(Xt_s[sel] + PERTURB_MAG * bu)
    dm = pred_at(Xt_s[sel] - PERTURB_MAG * bu)
    eff = float(np.mean(np.abs(dp - dm)))
    null = []
    for _ in range(N_PERTURB_DIRS):
        v = rng_global.normal(size=len(bu)); v /= np.linalg.norm(v)
        null.append(float(np.mean(np.abs(pred_at(Xt_s[sel] + PERTURB_MAG * v)
                                         - pred_at(Xt_s[sel] - PERTURB_MAG * v)))))
    null = np.array(null)
    perc = float((null < eff).mean())
    check2 = bool(perc >= 0.95)

    # how much weight did beta put on the new features?
    w_disag = float(np.sum(np.abs(beta[nb:])) / (np.sum(np.abs(beta)) + 1e-12))

    print(f"  {h}:  fit r={r_fit:+.3f}  held-out CAL r={r_cal:+.3f}  "
          f"angle(PC1)={angs[0]:5.1f}  perturb pct={perc:.3f}  "
          f"disag weight={100*w_disag:.1f}%  -> "
          f"{'VALIDATED' if (check1 and check2) else 'NOT VALIDATED'}")

    rows.append(dict(horizon=h, n_fit=len(fit_ok), n_cal=len(cal_ok),
                     n_features=Xf.shape[1],
                     ridge_alpha=round(float(reg.alpha_), 6),
                     r_fit=round(float(r_fit), 4),
                     r_cal_heldout=round(float(r_cal), 4),
                     rho_cal_heldout=round(float(rho_cal), 4),
                     angle_pc1_deg=round(angs[0], 2),
                     angle_pc2_deg=round(angs[1], 2),
                     angle_pc3_deg=round(angs[2], 2),
                     check1_not_variance=check1,
                     perturb_effect_beta=round(eff, 6),
                     perturb_null_p95=round(float(np.percentile(null, 95)), 6),
                     perturb_percentile=round(perc, 4),
                     check2_perturbation=check2,
                     disag_weight_frac=round(w_disag, 4),
                     beta_validated=bool(check1 and check2)))

V = pd.DataFrame(rows)
V.to_csv(f"{OUT}/task23_2_validation.csv", index=False)

print("\n" + "=" * 100)
print("FULL GATE TABLE (all four horizons, both checks) — Task 23 vs Task 21")
print("=" * 100)
old = pd.read_csv(f"{OUT}/task21_1_beta.csv").set_index("horizon")
print(f"{'horizon':8} {'r_cal(21)':>10} {'r_cal(23)':>10} "
      f"{'pct(21)':>9} {'pct(23)':>9} {'valid(21)':>10} {'valid(23)':>10}")
for r in V.itertuples():
    o = old.loc[r.horizon]
    print(f"{r.horizon:8} {o.r_cal_heldout:10.4f} {r.r_cal_heldout:10.4f} "
          f"{o.perturb_percentile:9.3f} {r.perturb_percentile:9.3f} "
          f"{str(bool(o.beta_validated)):>10} {str(r.beta_validated):>10}")
print(f"\n  Task 21: {int(old.beta_validated.sum())}/4 validated   "
      f"Task 23: {int(V.beta_validated.sum())}/4 validated")
print(f"Saved {OUT}/task23_2_validation.csv")
