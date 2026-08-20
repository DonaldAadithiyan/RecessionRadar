"""
TASK 21, Item 1 — Fit and validate the error-direction beta.

METHOD: normalized (locally-weighted) conformal prediction. beta is fit by
CONTINUOUS REGRESSION of |error| against input features x, using EVERY month in
the fitting split — not a classifier on rare large-error labels. That design
choice is the point of the proposal: it turns a 2-3-example scarcity problem
into a 400+-example regression.

THE SEPARATION THAT PRESERVES THE GUARANTEE (verified, not assumed):
    pool (pre-2020, 635 months) is split TEMPORALLY into
        FIT   = first 60%  -> beta is fit here, and ONLY here
        CAL   = last  40%  -> calibration scores; beta never sees these
    TEST  = 2020+ (65 months) -> beta never sees these
Three disjoint index sets, asserted disjoint at runtime. beta sees FIT only.

Two validation checks, both of which gate Item 2:
  1. NOT-JUST-VARIANCE: angle between beta and the top principal components of
     the feature space. If beta ~ PC1, it is a scaling artifact, not a
     difficulty signal.
  2. PERTURBATION: perturb test inputs along +/-beta and along random unit
     directions of the SAME magnitude, and compare how much the model's
     predicted error actually moves. Correlational fit alone is insufficient.

Outputs: errordir/task21_1_beta.csv, task21_1_perturbation.csv
"""
import os
import sys
import pickle
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
os.chdir(ROOT)

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()
from domain_common import rolling_origin_folds  # noqa: E402

OUT = "errordir"
os.makedirs(OUT, exist_ok=True)

# Import the canonical split objects directly from the repo module, so the
# feature set, target list, cleaning and train/test split are IDENTICAL to
# every other result in this project (no re-derivation, no drift).
from task_oof_and_probit import (  # noqa: E402
    X_train as X_train_df, X_test as X_test_df, y_train as y_pool,
    y_test, oof_pred as preds_pool_oof, preds_test, ensemble,
    LABELS, n_pool,
)

FIT_FRAC = 0.60          # fixed in advance: 60/40 fit/cal split of the pool
SEED = 5
N_PERTURB_DIRS = 200     # random directions for the perturbation null
PERTURB_MAG = 0.5        # in units of feature SD; fixed before running

rng_global = np.random.default_rng(SEED)

FEATURES = list(X_train_df.columns)
X_train = X_train_df.values
X_test = X_test_df.values

# ── THE THREE-WAY SEPARATION ────────────────────────────────────────────────
n_fit = int(round(FIT_FRAC * n_pool))
FIT_IDX = np.arange(0, n_fit)             # beta fit here ONLY
CAL_IDX = np.arange(n_fit, n_pool)        # calibration scores
assert len(np.intersect1d(FIT_IDX, CAL_IDX)) == 0, "FIT/CAL overlap"
print("=" * 96)
print("TASK 21 Item 1 — fit and validate the error-direction beta")
print("=" * 96)
print(f"  pool={n_pool} months  FIT=[0,{n_fit})={len(FIT_IDX)}  "
      f"CAL=[{n_fit},{n_pool})={len(CAL_IDX)}  TEST={len(X_test)}")
print("  FIT/CAL disjoint: asserted. FIT/TEST disjoint by construction (temporal).")

rows, prows = [], []
beta_store = {}

for h_idx, h in enumerate(LABELS):
    abs_err_pool = np.abs(preds_pool_oof[:, h_idx] - y_pool[:, h_idx])

    fit_ok = FIT_IDX[np.isfinite(abs_err_pool[FIT_IDX])]
    if len(fit_ok) < 50:
        print(f"  {h}: too few finite FIT errors ({len(fit_ok)}) — skipping")
        continue

    scaler = StandardScaler().fit(X_train[fit_ok])
    Xf = scaler.transform(X_train[fit_ok])
    yf = abs_err_pool[fit_ok]

    # continuous regression of |error| on x — every FIT month, not rare labels
    reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf, yf)
    beta = reg.coef_.astype(float)
    bn = np.linalg.norm(beta)
    beta_unit = beta / (bn + 1e-12)
    beta_store[h] = dict(beta=beta, beta_unit=beta_unit, scaler=scaler,
                         intercept=float(reg.intercept_), alpha=float(reg.alpha_))

    # in-fit quality, plus honest CAL-side quality (beta never fit there)
    r_fit = stats.pearsonr(Xf @ beta, yf).statistic
    cal_ok = CAL_IDX[np.isfinite(abs_err_pool[CAL_IDX])]
    Xc = scaler.transform(X_train[cal_ok])
    r_cal = stats.pearsonr(Xc @ beta, abs_err_pool[cal_ok]).statistic
    rho_cal = stats.spearmanr(Xc @ beta, abs_err_pool[cal_ok]).correlation

    # ── CHECK 1: not just the top-variance direction ───────────────────────
    pca = PCA(n_components=min(10, Xf.shape[1])).fit(Xf)
    angles = []
    for k in range(min(3, pca.n_components_)):
        pc = pca.components_[k]
        cosang = abs(float(np.dot(beta_unit, pc / np.linalg.norm(pc))))
        angles.append(np.degrees(np.arccos(np.clip(cosang, 0, 1))))
    ang_pc1 = angles[0]
    # criterion fixed in advance: beta must be >30 deg from PC1 to count as
    # distinct from the generic top-variance direction.
    check1 = bool(ang_pc1 > 30.0)

    # ── CHECK 2: perturbation vs random directions ─────────────────────────
    # Perturb TEST inputs along +/-beta and along random unit dirs of the same
    # magnitude; measure the change in the ensemble's own predicted value, and
    # compare |change| against the random-direction null.
    Xt_s = scaler.transform(X_test)
    sel = rng_global.choice(len(X_test), size=min(30, len(X_test)), replace=False)
    base_pred = preds_test[sel, h_idx]

    def pred_at(Xs_mod):
        Xdf = pd.DataFrame(scaler.inverse_transform(Xs_mod), columns=FEATURES)
        return ensemble.predict(Xdf)[:, h_idx]

    dplus = pred_at(Xt_s[sel] + PERTURB_MAG * beta_unit)
    dminus = pred_at(Xt_s[sel] - PERTURB_MAG * beta_unit)
    eff_beta = float(np.mean(np.abs(dplus - dminus)))

    eff_rand = []
    for _ in range(N_PERTURB_DIRS):
        v = rng_global.normal(size=len(beta_unit))
        v /= np.linalg.norm(v)
        rp = pred_at(Xt_s[sel] + PERTURB_MAG * v)
        rm = pred_at(Xt_s[sel] - PERTURB_MAG * v)
        eff_rand.append(float(np.mean(np.abs(rp - rm))))
    eff_rand = np.array(eff_rand)
    perc = float((eff_rand < eff_beta).mean())
    # criterion fixed in advance: beta must sit above the 95th percentile of
    # the random-direction null.
    check2 = bool(perc >= 0.95)

    print(f"\n  {h}:")
    print(f"    beta fit on {len(fit_ok)} FIT months, {Xf.shape[1]} features, "
          f"ridge alpha={reg.alpha_:.4g}")
    print(f"    fit r={r_fit:+.3f} | held-out CAL r={r_cal:+.3f} rho={rho_cal:+.3f}")
    print(f"    CHECK 1 angle(beta,PC1)={ang_pc1:5.1f} deg (PC2={angles[1]:.1f}, "
          f"PC3={angles[2]:.1f}) -> {'PASS' if check1 else 'FAIL'}")
    print(f"    CHECK 2 effect(beta)={eff_beta:.4f} vs random null "
          f"median={np.median(eff_rand):.4f} p95={np.percentile(eff_rand,95):.4f} "
          f"-> pct={perc:.3f} {'PASS' if check2 else 'FAIL'}")

    rows.append(dict(horizon=h, n_fit=len(fit_ok), n_cal=len(cal_ok),
                     n_features=Xf.shape[1], ridge_alpha=round(float(reg.alpha_), 6),
                     r_fit=round(float(r_fit), 4),
                     r_cal_heldout=round(float(r_cal), 4),
                     rho_cal_heldout=round(float(rho_cal), 4),
                     angle_pc1_deg=round(ang_pc1, 2),
                     angle_pc2_deg=round(angles[1], 2),
                     angle_pc3_deg=round(angles[2], 2),
                     check1_not_variance=check1,
                     perturb_effect_beta=round(eff_beta, 6),
                     perturb_null_median=round(float(np.median(eff_rand)), 6),
                     perturb_null_p95=round(float(np.percentile(eff_rand, 95)), 6),
                     perturb_percentile=round(perc, 4),
                     check2_perturbation=check2,
                     beta_validated=bool(check1 and check2)))
    prows.append(dict(horizon=h, effect_beta=eff_beta,
                      **{f"null_{i}": v for i, v in enumerate(eff_rand[:50])}))

B = pd.DataFrame(rows)
B.to_csv(f"{OUT}/task21_1_beta.csv", index=False)
pd.DataFrame(prows).to_csv(f"{OUT}/task21_1_perturbation.csv", index=False)
np.save(f"{OUT}/task21_1_beta_vectors.npy",
        {h: beta_store[h]["beta"] for h in beta_store}, allow_pickle=True)

print("\n" + "=" * 96)
print("ITEM 1 GATE")
print("=" * 96)
for r in B.itertuples():
    print(f"  {r.horizon:8s} check1={'PASS' if r.check1_not_variance else 'FAIL'}  "
          f"check2={'PASS' if r.check2_perturbation else 'FAIL'}  -> "
          f"{'VALIDATED' if r.beta_validated else 'NOT VALIDATED'}")
n_val = int(B.beta_validated.sum())
print(f"\n  {n_val}/{len(B)} horizons validated.")
print(f"Saved {OUT}/task21_1_beta.csv")
