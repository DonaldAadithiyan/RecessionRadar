"""
TASK 37, Item 1 — is Check 2's ridge PASS construction-trivial?

HYPOTHESIS: beta is fit by regression against |error| using the input features.
For a linear model, any regression-fit direction tends to align with the model's
own sensitivity geometry, so perturbing along it produces a larger output change
than a random direction -- regardless of whether it says anything about WHERE
errors are large. If so, Check 2's 0.995-1.000 ridge passes measure "this
direction was fitted" rather than "this direction is difficulty-relevant."

TEST: run the IDENTICAL Check 2 on directions fit the SAME way but to targets
that carry no error information:
    beta          RidgeCV(features -> |error|)          the real direction
    noise_target  RidgeCV(features -> N(0,1) noise)     fit, but meaningless
    perm_target   RidgeCV(features -> shuffled |error|) fit, label-permuted
    random_unit   an untrained unit vector              reference point

If noise_target / perm_target also score ~0.99, Check 2's ridge pass is an
artifact of fitting, not evidence about beta.

Check 2 implementation is copied unchanged from task34_3_retest.py:
  percentile = fraction of N random unit directions whose mean induced change
  is below the candidate's mean induced change.

Outputs: errordir/task37_1_check2_triviality.csv
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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

OUT = "errordir"
FIT_FRAC, SEED = 0.60, 5
N_DIRS, MAG, N_POINTS, N_TEST, POOL_CAP = 200, 0.5, 400, 400, 4000
N_REPS = 10          # repeats for the stochastic candidate directions

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


rows = []
for dname in DATA:
    Xp, yp, Xt, yt = load(dname)
    n_pool = len(yp); n_fit = int(round(FIT_FRAC * n_pool))
    FIT_I = np.arange(0, n_fit)

    sc0 = StandardScaler().fit(Xp)
    A = sc0.transform(Xp)
    oof = np.full(n_pool, np.nan)
    for tr, te in rolling_origin_folds(n_pool, n_folds=5):
        oof[te] = Ridge(alpha=1.0).fit(A[tr], yp[tr]).predict(A[te])
    mfull = Ridge(alpha=1.0).fit(A, yp)
    scores = np.abs(oof - yp); ok = np.isfinite(scores)
    fit_ok = FIT_I[ok[FIT_I]]

    scb = StandardScaler().fit(Xp[fit_ok])
    Zf, ef = scb.transform(Xp[fit_ok]), scores[fit_ok]
    Zt = scb.transform(Xt)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(Zt), size=min(N_POINTS, len(Zt)), replace=False)
    X0 = Zt[sel]

    def pred(Zs):
        raw = scb.inverse_transform(Zs)
        return mfull.predict(sc0.transform(raw))

    def mean_induced(v):
        V = np.repeat(np.atleast_2d(v), len(X0), axis=0)
        return float(np.mean(np.abs(pred(X0 + MAG * V) - pred(X0 - MAG * V))))

    def fit_dir(target):
        c = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Zf, target).coef_
        return c / (np.linalg.norm(c) + 1e-12)

    # the Check 2 null, unmodified
    dirs = rng.normal(size=(N_DIRS, Zf.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    null = np.array([mean_induced(u) for u in dirs])

    def check2(v):
        return float((null < mean_induced(v)).mean())

    # candidates
    cands = {"beta (real)": [fit_dir(ef)]}
    noise_ds, perm_ds, rand_ds = [], [], []
    for r in range(N_REPS):
        rr = np.random.default_rng(1000 + r)
        noise_ds.append(fit_dir(rr.normal(size=len(ef))))
        perm_ds.append(fit_dir(rr.permutation(ef)))
        u = rr.normal(size=Zf.shape[1]); rand_ds.append(u / np.linalg.norm(u))
    cands["noise_target (fitted)"] = noise_ds
    cands["perm_target (fitted)"] = perm_ds
    cands["random_unit (untrained)"] = rand_ds

    for cname, vs in cands.items():
        pcts = [check2(v) for v in vs]
        print(f"  {dname:10s} {cname:26s} Check2 pct: "
              f"mean={np.mean(pcts):.3f}  min={np.min(pcts):.3f}  "
              f"max={np.max(pcts):.3f}  pass_rate={np.mean(np.array(pcts)>=0.95):.2f}"
              f"  (n={len(pcts)})")
        rows.append(dict(domain=dname, candidate=cname, n=len(pcts),
                         check2_mean=round(float(np.mean(pcts)), 4),
                         check2_min=round(float(np.min(pcts)), 4),
                         check2_max=round(float(np.max(pcts)), 4),
                         pass_rate=round(float(np.mean(np.array(pcts) >= 0.95)), 3)))

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task37_1_check2_triviality.csv", index=False)
print("\n" + "=" * 96)
print("ITEM 1 — does Check 2 pass on ridge for ANY fitted direction?")
print("=" * 96)
print(D.to_string(index=False))
print(f"\nSaved {OUT}/task37_1_check2_triviality.csv")
