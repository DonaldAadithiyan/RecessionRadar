"""
TASK 24, Items 2 & 3 — fit beta independently for four domain-model pairs, and
re-run Task 21's full validation gate fresh on each.

Four fits: healthcare-ridge, healthcare-gradboost, climate-ridge,
climate-gradboost. Recession's beta is NEVER transferred. No pooling.

Per-fit protocol, identical to task21_1_beta.py:
  - three-way split of that domain's OWN pool: FIT 60% / CAL 40%, TEST separate
  - continuous RidgeCV regression of |error| on features, every FIT unit
  - CHECK 1: angle(beta, PC1) > 30 degrees
  - CHECK 2: perturbation effect >= 95th pct of a 200-direction random null

Features = that domain's base features + 1 disagreement feature
(|pred_ridge - pred_gb|), per the Item 1 audit. Disagreement is computed OOF on
the pool and from the full-fit models on test, matching how each domain already
produces its own predictions.

Outputs: errordir/task24_3_validation.csv
"""
import os
import sys
import warnings
import contextlib
import io
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
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
from domain_common import rolling_origin_folds  # noqa: E402

OUT = "errordir"
FIT_FRAC, SEED = 0.60, 5
N_PERTURB_DIRS, PERTURB_MAG = 200, 0.5     # identical to Task 21
rng_global = np.random.default_rng(SEED)

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import domain_healthcare as HC
    import domain_climate as CL

DOMAINS = {
    "healthcare": dict(mod=HC, Xpool=HC.X_pool, Xtest=HC.X_test,
                       ypool=HC.y_pool_pct, ytest=HC.Y_TEST),
    "climate": dict(mod=CL, Xpool=CL.X_pool, Xtest=CL.X_test,
                    ypool=CL.y_pool, ytest=CL.Y_TEST),
}


def disagreement_features(d):
    """
    |pred_ridge - pred_gb|, computed OOF on the pool and from full-fit models on
    test — the same construction each domain already uses for its own scores.
    Uses X only; no outcome enters.
    """
    Xp, yp, Xt = d["Xpool"], d["ypool"], d["Xtest"]
    sc = StandardScaler().fit(Xp)
    Xp_s, Xt_s = sc.transform(Xp), sc.transform(Xt)
    oof = {}
    tst = {}
    for name in ["ridge", "gradboost"]:
        A = Xp_s if name == "ridge" else Xp
        Bx = Xt_s if name == "ridge" else Xt
        o = np.full(len(yp), np.nan)
        for tr, te in rolling_origin_folds(len(yp), n_folds=5):
            ok = tr[np.isfinite(yp[tr])]
            if len(ok) < 20:
                continue
            m = (Ridge(alpha=1.0) if name == "ridge"
                 else HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                                    max_depth=4, random_state=SEED))
            m.fit(A[ok], yp[ok]); o[te] = m.predict(A[te])
        oof[name] = o
        mf = (Ridge(alpha=1.0) if name == "ridge"
              else HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                                 max_depth=4, random_state=SEED))
        okf = np.isfinite(yp)
        mf.fit(A[okf], yp[okf]); tst[name] = mf.predict(Bx)
    return (np.abs(oof["ridge"] - oof["gradboost"]),
            np.abs(tst["ridge"] - tst["gradboost"]))


print("=" * 104)
print("TASK 24 Items 2-3 — four independent beta fits + fresh validation gate")
print("=" * 104)

rows = []
STORE = {}
for dname, d in DOMAINS.items():
    disag_pool, disag_test = disagreement_features(d)
    for model in ["ridge", "gradboost"]:
        scores = d["mod"].SCORES[model]
        preds_test = d["mod"].PREDS_TEST[model]
        Xp, Xt = d["Xpool"], d["Xtest"]

        Xp_aug = np.column_stack([Xp, disag_pool])
        Xt_aug = np.column_stack([Xt, disag_test])

        n_pool = len(scores)
        n_fit = int(round(FIT_FRAC * n_pool))
        FIT_IDX, CAL_IDX = np.arange(0, n_fit), np.arange(n_fit, n_pool)
        assert len(np.intersect1d(FIT_IDX, CAL_IDX)) == 0

        ok = np.isfinite(scores) & np.isfinite(Xp_aug).all(axis=1)
        fit_ok = FIT_IDX[ok[FIT_IDX]]
        cal_ok = CAL_IDX[ok[CAL_IDX]]
        if len(fit_ok) < 50:
            print(f"  {dname}/{model}: too few FIT units ({len(fit_ok)}) — skipped")
            continue

        sc = StandardScaler().fit(Xp_aug[fit_ok])
        Xf = sc.transform(Xp_aug[fit_ok])
        reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf, scores[fit_ok])
        beta = reg.coef_.astype(float)
        bu = beta / (np.linalg.norm(beta) + 1e-12)

        r_fit = stats.pearsonr(Xf @ beta, scores[fit_ok]).statistic
        Xc = sc.transform(Xp_aug[cal_ok])
        r_cal = stats.pearsonr(Xc @ beta, scores[cal_ok]).statistic
        rho_cal = stats.spearmanr(Xc @ beta, scores[cal_ok]).correlation

        # CHECK 1
        pca = PCA(n_components=min(10, Xf.shape[1])).fit(Xf)
        angs = [np.degrees(np.arccos(np.clip(abs(float(np.dot(
            bu, pca.components_[k] / np.linalg.norm(pca.components_[k])))), 0, 1)))
            for k in range(3)]
        check1 = bool(angs[0] > 30.0)

        # CHECK 2 — perturb along beta vs random dirs, measure change in the
        # scored model's own prediction. Model refit once on FIT for this probe.
        nb = Xp.shape[1]
        base_m = (Ridge(alpha=1.0) if model == "ridge"
                  else HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                                     max_depth=4, random_state=SEED))
        yfit = d["ypool"][fit_ok]
        tgt_ok = np.isfinite(yfit)
        sc_base = StandardScaler().fit(Xp[fit_ok])
        _Xb = sc_base.transform(Xp[fit_ok]) if model == "ridge" else Xp[fit_ok]
        base_m.fit(_Xb[tgt_ok], yfit[tgt_ok])
        Xt_s_aug = sc.transform(Xt_aug)
        sel = rng_global.choice(len(Xt_aug), size=min(40, len(Xt_aug)), replace=False)

        def pred_at(Xs_mod):
            full = sc.inverse_transform(Xs_mod)[:, :nb]
            return base_m.predict(sc_base.transform(full) if model == "ridge" else full)

        dp = pred_at(Xt_s_aug[sel] + PERTURB_MAG * bu)
        dm = pred_at(Xt_s_aug[sel] - PERTURB_MAG * bu)
        eff = float(np.mean(np.abs(dp - dm)))
        null = []
        for _ in range(N_PERTURB_DIRS):
            v = rng_global.normal(size=len(bu)); v /= np.linalg.norm(v)
            null.append(float(np.mean(np.abs(pred_at(Xt_s_aug[sel] + PERTURB_MAG * v)
                                             - pred_at(Xt_s_aug[sel] - PERTURB_MAG * v)))))
        null = np.array(null)
        perc = float((null < eff).mean())
        check2 = bool(perc >= 0.95)

        w_disag = float(abs(beta[-1]) / (np.sum(np.abs(beta)) + 1e-12))
        key = f"{dname}/{model}"
        STORE[key] = dict(beta=beta, scaler=sc, fit_ok=fit_ok, cal_ok=cal_ok,
                          Xp_aug=Xp_aug, Xt_aug=Xt_aug, scores=scores,
                          preds_test=preds_test, ytest=d["ytest"])

        print(f"  {key:24s} n_fit={len(fit_ok):5d} n_cal={len(cal_ok):5d}  "
              f"fit r={r_fit:+.3f}  CAL r={r_cal:+.3f}  angle={angs[0]:5.1f}  "
              f"pct={perc:.3f}  disag_w={100*w_disag:4.1f}%  -> "
              f"{'VALIDATED' if (check1 and check2) else 'NOT VALIDATED'}")

        rows.append(dict(domain=dname, model=model, n_fit=len(fit_ok),
                         n_cal=len(cal_ok), n_features=Xf.shape[1],
                         ridge_alpha=round(float(reg.alpha_), 6),
                         r_fit=round(float(r_fit), 4),
                         r_cal_heldout=round(float(r_cal), 4),
                         rho_cal_heldout=round(float(rho_cal), 4),
                         angle_pc1_deg=round(angs[0], 2),
                         angle_pc2_deg=round(angs[1], 2),
                         check1_not_variance=check1,
                         perturb_effect=round(eff, 6),
                         perturb_null_p95=round(float(np.percentile(null, 95)), 6),
                         perturb_percentile=round(perc, 4),
                         check2_perturbation=check2,
                         disag_weight_frac=round(w_disag, 4),
                         beta_validated=bool(check1 and check2)))

V = pd.DataFrame(rows)
V.to_csv(f"{OUT}/task24_3_validation.csv", index=False)
np.save(f"{OUT}/task24_store.npy", {k: {kk: vv for kk, vv in v.items()}
                                    for k, v in STORE.items()}, allow_pickle=True)
print("\n" + "=" * 104)
print("ITEM 3 GATE — full four-row table")
print("=" * 104)
print(V[["domain", "model", "r_cal_heldout", "angle_pc1_deg", "perturb_percentile",
         "check1_not_variance", "check2_perturbation", "beta_validated"]].to_string(index=False))
print(f"\n  validated: {int(V.beta_validated.sum())}/{len(V)}")
print(f"Saved {OUT}/task24_3_validation.csv")
