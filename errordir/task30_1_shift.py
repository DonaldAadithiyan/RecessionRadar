"""
TASK 30, Item 1 — is energy's regime change COVARIATE SHIFT or CONCEPT DRIFT?

RLCP's guarantee is specifically for covariate shift: P(X) changes while P(Y|X)
holds. This tests which one energy actually has. Diagnostic only -- no interval
construction.

TEST 1 — does P(X) shift?
  Per-feature two-sample tests (KS) between FIT-period and TEST-period marginals,
  plus a multivariate check: train a logistic classifier to discriminate FIT rows
  from TEST rows on features alone. AUC ~0.5 means no detectable covariate shift;
  AUC >> 0.5 means P(X) genuinely differs.

TEST 2 — does P(Y|X) hold?
  The practical proxy the task specifies: P(error | z). If the feature->outcome
  RELATIONSHIP is stable, then conditional on the same difficulty coordinate z,
  the error distribution should look the same in FIT and TEST. Measured three
  ways:
    (a) per-z-bin mean/quantile of |error|, FIT vs TEST, on MATCHED z bins
    (b) KS test on |error| within each matched z bin
    (c) a direct regression check: does the fitted score->z relationship from FIT
        still predict TEST errors at the same level?

WHY (a)-(c) TOGETHER: a raw comparison of error distributions would conflate the
two — if P(X) shifts toward harder inputs, TEST errors get larger even with a
perfectly stable P(Y|X). Conditioning on z is what separates them: matched z
means matched predicted difficulty, so a remaining difference is drift in the
relationship itself.

Climate is run alongside as a CONTROL: Task 29 found all methods tie there and
nothing breaks, so climate should show materially less shift on both tests. If
climate looks the same as energy, the diagnostic is not discriminating.

Outputs: errordir/task30_1_shift.csv
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
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

OUT = "errordir"
FIT_FRAC = 0.60

_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    import domain_climate as CL
EN = pd.read_csv("data/domains/task25/energy.csv")


def get_domain(name):
    if name == "climate":
        return CL.X_pool, CL.y_pool, CL.X_test, np.asarray(CL.Y_TEST, float)
    y = EN["target"].values.astype(float)
    X = EN.drop(columns=["target"]).values.astype(float)
    keep = np.arange(max(0, len(y) - 4400), len(y))
    X, y = X[keep], y[keep]
    return X[:-400], y[:-400], X[-400:], y[-400:]


rows = []
for dom in ["energy", "climate"]:
    Xp, yp, Xt, yt = get_domain(dom)
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

    # ── TEST 1: does P(X) shift? ───────────────────────────────────────────
    Xf, Xte = Xp[fit_ok], Xt
    ks = [stats.ks_2samp(Xf[:, j], Xte[:, j]).statistic for j in range(Xf.shape[1])]
    ks_p = [stats.ks_2samp(Xf[:, j], Xte[:, j]).pvalue for j in range(Xf.shape[1])]
    frac_sig = float(np.mean(np.array(ks_p) < 0.01))
    # multivariate: can a classifier tell FIT from TEST on features alone?
    Xc = np.vstack([Xf, Xte]); yc = np.r_[np.zeros(len(Xf)), np.ones(len(Xte))]
    scc = StandardScaler().fit(Xc)
    auc = float(np.mean(cross_val_score(
        LogisticRegression(max_iter=2000), scc.transform(Xc), yc,
        cv=5, scoring="roc_auc")))

    # ── TEST 2: does P(error | z) hold? ───────────────────────────────────
    # MATCHED z bins defined on the FIT distribution, applied to both.
    edges = np.quantile(z_fit, [0, .2, .4, .6, .8, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    bin_rows, ks_bin_p, ratios = [], [], []
    for i in range(5):
        mf = (z_fit >= edges[i]) & (z_fit < edges[i + 1])
        mt = (z_test >= edges[i]) & (z_test < edges[i + 1])
        if mf.sum() < 10 or mt.sum() < 10:
            continue
        ef, et = err_fit[mf], err_test[mt]
        p = float(stats.ks_2samp(ef, et).pvalue)
        ks_bin_p.append(p)
        ratios.append(float(np.median(et) / max(np.median(ef), 1e-9)))
        bin_rows.append((i + 1, int(mf.sum()), int(mt.sum()),
                         float(np.median(ef)), float(np.median(et)), p))

    frac_bins_drift = float(np.mean(np.array(ks_bin_p) < 0.01)) if ks_bin_p else np.nan
    med_ratio = float(np.median(ratios)) if ratios else np.nan

    print(f"\n{'='*92}\n{dom.upper()}\n{'='*92}")
    print(f"  TEST 1 — P(X) shift")
    print(f"    per-feature KS: median D={np.median(ks):.3f}  "
          f"fraction with p<0.01: {frac_sig:.2f} of {len(ks)} features")
    print(f"    FIT-vs-TEST classifier AUC (features only): {auc:.3f}  "
          f"({'strong covariate shift' if auc > 0.8 else 'moderate' if auc > 0.65 else 'weak/none'})")
    print(f"  TEST 2 — P(error | z) stability, on MATCHED z bins")
    print(f"    {'bin':>4} {'n_fit':>7} {'n_test':>7} {'med|e|FIT':>10} {'med|e|TEST':>11} {'KS p':>9}")
    for b in bin_rows:
        print(f"    {b[0]:>4} {b[1]:>7} {b[2]:>7} {b[3]:>10.3f} {b[4]:>11.3f} {b[5]:>9.4f}")
    print(f"    fraction of bins with KS p<0.01: {frac_bins_drift:.2f}")
    print(f"    median TEST/FIT error ratio within matched bins: {med_ratio:.3f}")

    rows.append(dict(domain=dom, n_fit=len(fit_ok), n_test=len(yt),
                     ks_median_D=round(float(np.median(ks)), 4),
                     frac_features_ks_sig=round(frac_sig, 3),
                     classifier_auc=round(auc, 4),
                     frac_zbins_error_drift=round(frac_bins_drift, 3),
                     median_test_fit_error_ratio=round(med_ratio, 3)))

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task30_1_shift.csv", index=False)
print("\n" + "=" * 92)
print("VERDICT")
print("=" * 92)
for r in D.itertuples():
    cov_shift = r.classifier_auc > 0.65
    concept = (r.frac_zbins_error_drift >= 0.4) or (r.median_test_fit_error_ratio > 1.5
                                                    or r.median_test_fit_error_ratio < 0.67)
    if cov_shift and not concept:
        v = "PURE COVARIATE SHIFT (P(X) moves, P(err|z) holds)"
    elif cov_shift and concept:
        v = "COVARIATE SHIFT *AND* CONCEPT DRIFT"
    elif concept:
        v = "CONCEPT DRIFT without much P(X) movement"
    else:
        v = "NEITHER strongly detected"
    print(f"  {r.domain:9s} AUC={r.classifier_auc:.3f}  "
          f"drifted z-bins={r.frac_zbins_error_drift:.2f}  "
          f"err ratio={r.median_test_fit_error_ratio:.2f}  -> {v}")
print(f"\nSaved {OUT}/task30_1_shift.csv")
