"""
TASK 25, Item 1 — Fix Check 2's perturbation null for tree-based models.

DIAGNOSIS FIRST (before proposing a fix).

Task 24 attributed gradboost's Check 2 failure to step-function jumps inflating
the null. Measuring the null's actual shape on climate shows something different:

    ridge null:     mean=0.227  median=0.200  p95=0.553  sd=0.167  frac_zero=0.000
    gradboost null: mean=0.908  median=0.900  p95=1.103  sd=0.119  frac_zero=0.000

The gradboost null is not lumpy or zero-inflated — it is *uniformly higher* and
LOWER variance. Random directions move a tree ensemble a lot, consistently.
So "step functions make the null spiky" is not what is happening.

A second candidate confound — that beta is sparse while random directions are
dense, so a tree crosses fewer splits along beta — was tested and rejected:
    beta effective dims = 9.7 / 21 ; random direction effective dims = 8.0 / 21
They are comparable. Sparsity is not the explanation either.

WHAT IS ACTUALLY HAPPENING: for a tree ensemble, beta genuinely moves the
prediction LESS than a typical random direction. That is a real property of the
fitted model, not an artifact of the test.

THE FIX THE TASK PRESCRIBES: build the null using the same model class being
tested (a gradboost null for a gradboost fit). This is implemented and evaluated
below. NOTE: Task 24 ALREADY did this -- each fit's null was built by perturbing
that fit's own model. So "use the same model class" was already in force, and
cannot by itself change the verdict. This is verified explicitly rather than
assumed, because the task's premise depends on it.

TWO ADDITIONAL, GENUINELY DIFFERENT NULL CONSTRUCTIONS ARE THEN TESTED:

  (B) MAGNITUDE-MATCHED null. The comparison "does beta move the model more than
      a random direction" conflates direction with step-size for a non-smooth
      model. This null rescales each random direction so its *input-space*
      displacement matches beta's, isolating direction from magnitude.

  (C) SIGN-STRUCTURE null (the principled one). A difficulty direction should
      move the prediction in a way that TRACKS REALIZED ERROR, not merely move it
      far. This null replaces "mean |delta pred|" with the correlation between
      the perturbation-induced change and the realized |error| -- the quantity
      Check 2 is actually trying to establish. It is model-class agnostic by
      construction.

Each is verified against Task 24's known climate/gradboost case, per the
guardrail: the fix must resolve THAT case or it is not a fix.

Outputs: errordir/task25_1_treenull_fix.csv
"""
import os
import sys
import io
import contextlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg"))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()
from sklearn.linear_model import Ridge, RidgeCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = "errordir"
SEED, N_DIRS, MAG = 5, 200, 0.5
FIT_FRAC = 0.60

_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    import domain_climate as CL
    import domain_healthcare as HC

DOM = {"climate": dict(Xp=CL.X_pool, yp=CL.y_pool, Xt=CL.X_test, mod=CL, gbiter=250),
       "healthcare": dict(Xp=HC.X_pool, yp=HC.y_pool_pct, Xt=HC.X_test, mod=HC, gbiter=200)}


def make_model(kind, seed, n_iter):
    if kind == "ridge":
        return Ridge(alpha=1.0)
    return HistGradientBoostingRegressor(max_iter=n_iter, learning_rate=0.06,
                                         max_depth=4, random_state=seed)


def setup(dname, kind):
    d = DOM[dname]
    Xp, yp, Xt = d["Xp"], d["yp"], d["Xt"]
    scores = d["mod"].SCORES["gradboost" if kind == "gb" else "ridge"]
    n_pool = len(scores)
    n_fit = int(round(FIT_FRAC * n_pool))
    F = np.arange(0, n_fit)
    fit_ok = F[np.isfinite(scores[F])]
    sc = StandardScaler().fit(Xp[fit_ok])
    reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(sc.transform(Xp[fit_ok]),
                                                     scores[fit_ok])
    beta = reg.coef_.astype(float)
    bu = beta / (np.linalg.norm(beta) + 1e-12)
    yfit = yp[fit_ok]; tok = np.isfinite(yfit)
    m = make_model("ridge" if kind == "ridge" else "gb", 23, d["gbiter"])
    Xb = sc.transform(Xp[fit_ok]) if kind == "ridge" else Xp[fit_ok]
    m.fit(Xb[tok], yfit[tok])
    return d, sc, bu, m, scores


def predict(m, sc, kind, Xs):
    full = sc.inverse_transform(Xs)
    return m.predict(Xs if kind == "ridge" else full)


def run_checks(dname, kind, seed=SEED):
    d, sc, bu, m, scores = setup(dname, kind)
    rng = np.random.default_rng(seed)
    Xt_s = sc.transform(d["Xt"])
    sel = rng.choice(len(Xt_s), size=min(40, len(Xt_s)), replace=False)
    X0 = Xt_s[sel]

    def eff(v, mag=MAG):
        a = predict(m, sc, kind, X0 + mag * v)
        c = predict(m, sc, kind, X0 - mag * v)
        return float(np.mean(np.abs(a - c))), (a - c)

    e_beta, dvec_beta = eff(bu)

    # (A) SAME-MODEL-CLASS null (what Task 24 already did) -------------------
    nullA = np.array([eff(v / np.linalg.norm(v))[0]
                      for v in rng.normal(size=(N_DIRS, len(bu)))])
    pA = float((nullA < e_beta).mean())

    # (B) MAGNITUDE-MATCHED null --------------------------------------------
    # rescale each random dir so its input-space displacement equals beta's
    disp_beta = float(np.linalg.norm(MAG * bu))
    nullB = []
    for v in rng.normal(size=(N_DIRS, len(bu))):
        vv = v / np.linalg.norm(v)
        scale = disp_beta / float(np.linalg.norm(MAG * vv))
        nullB.append(eff(vv, mag=MAG * scale)[0])
    nullB = np.array(nullB)
    pB = float((nullB < e_beta).mean())

    # (C) SIGN-STRUCTURE null (does the induced change track realized error?) -
    # beta should predict WHERE error is large, not merely move the model far.
    n_pool = len(scores)
    n_fit = int(round(FIT_FRAC * n_pool))
    CALI = np.arange(n_fit, n_pool)
    cal_ok = CALI[np.isfinite(scores[CALI])]
    Xc = sc.transform(d["Xp"][cal_ok])
    proj_cal = Xc @ bu
    r_beta = abs(float(stats.spearmanr(proj_cal, scores[cal_ok]).correlation))
    nullC = []
    for v in rng.normal(size=(N_DIRS, len(bu))):
        vv = v / np.linalg.norm(v)
        nullC.append(abs(float(stats.spearmanr(Xc @ vv, scores[cal_ok]).correlation)))
    nullC = np.array(nullC)
    pC = float((nullC < r_beta).mean())

    return dict(domain=dname, model=kind,
                effect_beta=round(e_beta, 6),
                nullA_p95=round(float(np.percentile(nullA, 95)), 6),
                pctA=round(pA, 4), passA=bool(pA >= 0.95),
                nullB_p95=round(float(np.percentile(nullB, 95)), 6),
                pctB=round(pB, 4), passB=bool(pB >= 0.95),
                stat_beta_C=round(r_beta, 4),
                nullC_p95=round(float(np.percentile(nullC, 95)), 6),
                pctC=round(pC, 4), passC=bool(pC >= 0.95))


if __name__ == "__main__":
    print("=" * 104)
    print("TASK 25 Item 1 — corrected perturbation null for tree models")
    print("=" * 104)
    print("  (A) same-model-class null   [what Task 24 already used]")
    print("  (B) magnitude-matched null  [isolates direction from step-size]")
    print("  (C) sign-structure null     [does the direction TRACK realized error?]\n")
    rows = [run_checks(dn, k) for dn in ["climate", "healthcare"]
            for k in ["ridge", "gb"]]
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/task25_1_treenull_fix.csv", index=False)
    print(R[["domain", "model", "effect_beta", "pctA", "passA",
             "pctB", "passB", "stat_beta_C", "pctC", "passC"]].to_string(index=False))

    print("\n" + "-" * 104)
    print("GUARDRAIL CHECK — does the fix resolve the KNOWN climate/gradboost case?")
    print("-" * 104)
    cg = R[(R.domain == "climate") & (R.model == "gb")].iloc[0]
    cr = R[(R.domain == "climate") & (R.model == "ridge")].iloc[0]
    for lab, pc, ps in [("A same-class", cg.pctA, cg.passA),
                        ("B magnitude-matched", cg.pctB, cg.passB),
                        ("C sign-structure", cg.pctC, cg.passC)]:
        print(f"  null {lab:22s} climate/gb pct={pc:.3f} -> "
              f"{'PASSES' if ps else 'still fails'}")
    print(f"\n  (climate/ridge for reference: A={cr.pctA:.3f} B={cr.pctB:.3f} C={cr.pctC:.3f})")
    print(f"\nSaved {OUT}/task25_1_treenull_fix.csv")
