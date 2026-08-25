"""
TASK 31, Item 1 — screen candidates for CONCEPT DRIFT using Task 30's criterion.

The diagnostic logic is Task 30's, applied unchanged:
  TEST 1  P(X) shift      -- FIT-vs-TEST classifier AUC on features alone
  TEST 2  P(err|z) drift  -- KS on |error| within MATCHED z bins (bins defined
                             on FIT), plus the median error ratio

QUALIFYING BAR (exactly energy's, no lowering):
    energy had frac_zbins_error_drift = 0.40 and median error ratio 1.315
    climate (the negative control) had 0.00 and 1.276
  A candidate qualifies only if it shows genuine drift in P(err|z) --
  operationally frac_zbins_error_drift >= 0.20 (at least 1 of 5 bins at p<0.01)
  AND a median error ratio outside [0.8, 1.25]. Covariate shift alone
  (high AUC, zero drifted bins) does NOT qualify -- that is the climate pattern.

EVERY candidate screened is reported, including rejected ones.

Outputs: errordir/task31_1_screening.csv
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
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

OUT = "errordir"
FIT_FRAC = 0.60
N_TEST = 400
POOL_CAP = 4000

CAND = {"fraud": "data/domains/task31/fraud.csv",
        "epidemic": "data/domains/task31/epidemic.csv"}


def load(path):
    d = pd.read_csv(path)
    if "date" in d.columns:
        d = d.sort_values("date").reset_index(drop=True)
        d = d.drop(columns=["date"])
    y = d["target"].values.astype(float)
    X = d.drop(columns=["target"]).values.astype(float)
    keep = np.arange(max(0, len(y) - (POOL_CAP + N_TEST)), len(y))  # keep time order
    return X[keep], y[keep]


rows = []
for name, path in CAND.items():
    if not os.path.exists(path):
        print(f"  {name}: NOT BUILT — skipped")
        continue
    X, y = load(path)
    Xp, yp, Xt, yt = X[:-N_TEST], y[:-N_TEST], X[-N_TEST:], y[-N_TEST:]
    n_pool = len(yp); n_fit = int(round(FIT_FRAC * n_pool))
    FIT_I = np.arange(0, n_fit)

    sc0 = StandardScaler().fit(Xp)
    okp = np.isfinite(yp)
    base_m = Ridge(alpha=1.0).fit(sc0.transform(Xp[okp]), yp[okp])
    oof = np.full(n_pool, np.nan)
    for tr, te in rolling_origin_folds(n_pool, n_folds=5):
        o = tr[np.isfinite(yp[tr])]
        if len(o) < 20: continue
        oof[te] = Ridge(alpha=1.0).fit(sc0.transform(Xp[o]), yp[o]).predict(sc0.transform(Xp[te]))
    scores = np.abs(oof - yp)
    fit_ok = FIT_I[np.isfinite(scores[FIT_I])]

    scb = StandardScaler().fit(Xp[fit_ok])
    bm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scb.transform(Xp[fit_ok]),
                                                    scores[fit_ok]).coef_
    bu = bm / (np.linalg.norm(bm) + 1e-12)
    z_fit = scb.transform(Xp[fit_ok]) @ bu
    z_test = scb.transform(Xt) @ bu
    err_fit = scores[fit_ok]
    err_test = np.abs(base_m.predict(sc0.transform(Xt)) - yt)

    # TEST 1
    Xf, Xte = Xp[fit_ok], Xt
    Xc = np.vstack([Xf, Xte]); yc = np.r_[np.zeros(len(Xf)), np.ones(len(Xte))]
    scc = StandardScaler().fit(Xc)
    auc = float(np.mean(cross_val_score(LogisticRegression(max_iter=2000),
                                        scc.transform(Xc), yc, cv=5,
                                        scoring="roc_auc")))
    # TEST 2
    edges = np.quantile(z_fit, [0, .2, .4, .6, .8, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    ks_p, ratios, brows = [], [], []
    for i in range(5):
        mf = (z_fit >= edges[i]) & (z_fit < edges[i + 1])
        mt = (z_test >= edges[i]) & (z_test < edges[i + 1])
        if mf.sum() < 10 or mt.sum() < 10: continue
        ef, et = err_fit[mf], err_test[mt]
        p = float(stats.ks_2samp(ef, et).pvalue)
        ks_p.append(p); ratios.append(float(np.median(et) / max(np.median(ef), 1e-9)))
        brows.append((i + 1, int(mf.sum()), int(mt.sum()),
                      float(np.median(ef)), float(np.median(et)), p))
    frac = float(np.mean(np.array(ks_p) < 0.01)) if ks_p else np.nan
    mr = float(np.median(ratios)) if ratios else np.nan
    qualifies = bool(frac >= 0.20 and (mr > 1.25 or mr < 0.8))

    print(f"\n{'='*88}\n{name.upper()}  (pool={n_pool}, test={len(yt)}, feats={Xp.shape[1]})\n{'='*88}")
    print(f"  TEST 1 classifier AUC = {auc:.3f}")
    print(f"  TEST 2 matched z-bins:")
    print(f"    {'bin':>4} {'n_fit':>7} {'n_test':>7} {'med|e|FIT':>10} {'med|e|TEST':>11} {'KS p':>9}")
    for b in brows:
        print(f"    {b[0]:>4} {b[1]:>7} {b[2]:>7} {b[3]:>10.4f} {b[4]:>11.4f} {b[5]:>9.4f}")
    print(f"    drifted bins = {frac:.2f}   median err ratio = {mr:.3f}")
    print(f"  -> {'QUALIFIES (concept drift)' if qualifies else 'REJECTED'}")

    rows.append(dict(candidate=name, n_pool=n_pool, n_test=len(yt),
                     n_features=Xp.shape[1], classifier_auc=round(auc, 4),
                     frac_zbins_error_drift=round(frac, 3),
                     median_test_fit_error_ratio=round(mr, 3),
                     qualifies=qualifies))

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task31_1_screening.csv", index=False)
print("\n" + "=" * 88)
print("SCREENING SUMMARY (bar: drifted bins >= 0.20 AND err ratio outside [0.8,1.25])")
print("=" * 88)
ref = pd.DataFrame([dict(candidate="energy (reference, qualified)", classifier_auc=0.799,
                         frac_zbins_error_drift=0.40, median_test_fit_error_ratio=1.315,
                         qualifies=True),
                    dict(candidate="climate (reference, control)", classifier_auc=0.958,
                         frac_zbins_error_drift=0.00, median_test_fit_error_ratio=1.276,
                         qualifies=False)])
print(pd.concat([D, ref], ignore_index=True)[
    ["candidate", "classifier_auc", "frac_zbins_error_drift",
     "median_test_fit_error_ratio", "qualifies"]].to_string(index=False))
print(f"\nSaved {OUT}/task31_1_screening.csv")
