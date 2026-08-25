"""
TASK 37, Items 2-4 — Held-Out Specificity (HOS), a linear-valid instrument.

THE QUESTION: does beta's held-out correlation with |error| exceed what a
direction fit by the SAME procedure to a meaningless target would achieve?

WHY THIS AVOIDS BOTH PRIOR FAILURE MODES:
  * It never computes a model-output derivative, so it cannot inherit DS's
    degeneracy (Task 36: delta_i constant because the ridge path is affine).
  * Its null consists of directions fit by the identical procedure, so it cannot
    inherit the "fitted vs untrained" geometric bias Item 1 was designed to
    detect (and which Item 1 found Check 2 does NOT suffer from, but which would
    reappear here if the null were untrained random vectors).

PROCEDURE (fully specified before application):
  1. Split the pool THREE ways: FIT (fit beta) / CAL / HOLDOUT.
     HOLDOUT is carved from the pool and used for NOTHING except scoring.
     It is disjoint from FIT and from CAL, and is never used to fit beta, to
     calibrate, or to select anything.
  2. beta = RidgeCV(features -> |error|) on FIT only.
  3. Statistic: HOS(v) = |Spearman( X_holdout @ v , |error|_holdout )|.
     Spearman because only ordering is claimed; absolute value because a
     difficulty direction is meaningful up to sign.
  4. NULL: for j = 1..N, draw a permutation pi_j of the FIT-split error labels,
     fit v_j = RidgeCV(features -> pi_j(|error|_fit)) using the IDENTICAL
     estimator, and compute HOS(v_j) on the SAME holdout slice.
     Permuting the labels preserves the target's marginal distribution and the
     fitting procedure exactly, destroying only the feature-error relationship.
  5. percentile = fraction of j with HOS(v_j) < HOS(beta); PASS iff >= 0.95.

  N = 200 null directions. HOLDOUT_FRAC = 0.25 of the pool.

VERIFICATION (Item 3) — both cases required before use on real fits:
  known-signal: linear-model data with |error| genuinely driven by a known
                direction. HOS must DETECT.
  known-null:   |error| independent of all features. HOS must REJECT.

Outputs: task37_3_synthetic.csv, task37_4_real.csv
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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

OUT = "errordir"
SEED = 37
N_NULL = 200
FIT_FRAC, HOLDOUT_FRAC = 0.60, 0.25
ALPHAS = np.logspace(-3, 3, 25)


def fit_dir(Z, t):
    c = RidgeCV(alphas=ALPHAS).fit(Z, t).coef_
    return c / (np.linalg.norm(c) + 1e-12)


def hos(v, Zh, eh):
    r = stats.spearmanr(Zh @ v, eh).correlation
    return abs(float(r)) if np.isfinite(r) else np.nan


def hos_test(Zf, ef, Zh, eh, n_null=N_NULL, seed=SEED):
    """Returns (HOS_beta, percentile, null_mean)."""
    b = fit_dir(Zf, ef)
    s = hos(b, Zh, eh)
    rng = np.random.default_rng(seed)
    ns = []
    for _ in range(n_null):
        vj = fit_dir(Zf, rng.permutation(ef))
        x = hos(vj, Zh, eh)
        if np.isfinite(x):
            ns.append(x)
    ns = np.array(ns)
    pct = float((ns < s).mean()) if len(ns) else np.nan
    return s, pct, float(np.mean(ns)) if len(ns) else np.nan


# ── ITEM 3: synthetic verification ─────────────────────────────────────────
def synth(kind, n=3000, d=10, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d); w[0] = 1.0
    f = 2.0 * X[:, 1] - 1.5 * X[:, 2]
    if kind == "signal":
        sd = 0.3 + 2.0 * np.maximum(X @ w, 0)      # error driven by direction w
    else:
        sd = np.full(n, 1.0)                        # error independent of X
    y = f + rng.normal(0, sd)
    return X, y


print("=" * 92)
print("ITEM 3 — synthetic verification of HOS (both cases required)")
print("=" * 92)
srows = []
for kind, must in [("signal", "DETECT"), ("null", "REJECT")]:
    X, y = synth(kind)
    n = len(y); n_fit = int(0.6 * n); n_hold = int(HOLDOUT_FRAC * n)
    fi = np.arange(0, n_fit)
    hi = np.arange(n - n_hold, n)
    assert len(np.intersect1d(fi, hi)) == 0
    sc = StandardScaler().fit(X[fi])
    m = Ridge(alpha=1.0).fit(sc.transform(X[fi]), y[fi])
    ef = np.abs(m.predict(sc.transform(X[fi])) - y[fi])
    eh = np.abs(m.predict(sc.transform(X[hi])) - y[hi])
    s, pct, nm = hos_test(sc.transform(X[fi]), ef, sc.transform(X[hi]), eh)
    det = bool(pct >= 0.95)
    ok = det if must == "DETECT" else not det
    print(f"  {kind:8s} (must {must})  HOS={s:.4f}  null_mean={nm:.4f}  "
          f"pct={pct:.3f}  -> {'DETECT' if det else 'REJECT'}  "
          f"{'PASS' if ok else 'FAIL'}")
    srows.append(dict(case=kind, requirement=must, HOS=round(s, 4),
                      null_mean=round(nm, 4), percentile=round(pct, 4),
                      verdict=("DETECT" if det else "REJECT"), correct=ok))
S = pd.DataFrame(srows)
S.to_csv(f"{OUT}/task37_3_synthetic.csv", index=False)
print(f"\n  ALL SYNTHETIC CASES CORRECT: {bool(S.correct.all())}")
if not bool(S.correct.all()):
    print("  -> instrument NOT verified; Item 4 must not run.")
    sys.exit(1)

# ── ITEM 4: the real fits ──────────────────────────────────────────────────
import io, contextlib
_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    import domain_climate as CL
EN = pd.read_csv("data/domains/task25/energy.csv")


def real(dname):
    if dname == "climate":
        return CL.X_pool, CL.y_pool
    y = EN["target"].values.astype(float)
    X = EN.drop(columns=["target"]).values.astype(float)
    keep = np.arange(max(0, len(y) - 4400), len(y))
    return X[keep][:-400], y[keep][:-400]


print("\n" + "=" * 92)
print("ITEM 4 — HOS on the two fits behind the positive claim")
print("=" * 92)
rrows = []
for dname in ["climate", "energy"]:
    Xp, yp = real(dname)
    n_pool = len(yp)
    n_fit = int(round(FIT_FRAC * n_pool))
    n_hold = int(round(HOLDOUT_FRAC * n_pool))
    FIT_I = np.arange(0, n_fit)
    HOLD_I = np.arange(n_pool - n_hold, n_pool)          # disjoint by construction
    assert len(np.intersect1d(FIT_I, HOLD_I)) == 0

    sc0 = StandardScaler().fit(Xp)
    A = sc0.transform(Xp)
    oof = np.full(n_pool, np.nan)
    for tr, te in rolling_origin_folds(n_pool, n_folds=5):
        o = tr[np.isfinite(yp[tr])]
        if len(o) < 20: continue
        oof[te] = Ridge(alpha=1.0).fit(A[o], yp[o]).predict(A[te])
    scores = np.abs(oof - yp); ok = np.isfinite(scores)
    fit_ok = FIT_I[ok[FIT_I]]; hold_ok = HOLD_I[ok[HOLD_I]]

    scb = StandardScaler().fit(Xp[fit_ok])
    Zf, ef = scb.transform(Xp[fit_ok]), scores[fit_ok]
    Zh, eh = scb.transform(Xp[hold_ok]), scores[hold_ok]
    s, pct, nm = hos_test(Zf, ef, Zh, eh)
    passed = bool(pct >= 0.95)
    print(f"  {dname:9s} n_fit={len(fit_ok):5d} n_hold={len(hold_ok):5d}  "
          f"HOS={s:.4f}  null_mean={nm:.4f}  pct={pct:.3f}  -> "
          f"{'VALIDATED' if passed else 'NOT VALIDATED'}")
    rrows.append(dict(domain=dname, model="ridge", mechanism="beta",
                      n_fit=len(fit_ok), n_holdout=len(hold_ok),
                      HOS_beta=round(s, 4), null_mean=round(nm, 4),
                      percentile=round(pct, 4), validated=passed))
R = pd.DataFrame(rrows)
R.to_csv(f"{OUT}/task37_4_real.csv", index=False)
print("\n" + R.to_string(index=False))
print(f"\nSaved {OUT}/task37_3_synthetic.csv, {OUT}/task37_4_real.csv")
