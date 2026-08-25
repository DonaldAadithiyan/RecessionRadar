"""
TASK 34, Item 2 — synthetic verification of the Directional Specificity (DS)
test, BOTH directions, before it is applied to any real fit.

The statistic is exactly as specified in task34_1_test_design.md. Nothing here
is adjusted based on results.

CASES (all required):
  A. KNOWN-SIGNAL   tree-like data with a genuine embedded difficulty direction.
                    DS must DETECT it (percentile >= 0.95).
  B. KNOWN-NULL     tree-like data, no difficulty direction, noise only.
                    DS must REJECT it (percentile < 0.95).
  C. KNOWN-NULL AT  same as B but with a hyper-responsive forecaster (deeper,
     HIGH RESPONSE  more trees), i.e. the exact condition that broke Check 2.
                    DS must still REJECT.

Check 2 is run alongside on all three, to show what it does under the same
conditions.

Outputs: errordir/task34_2_synthetic.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.join(HERE, ".."))

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

OUT = "errordir"
H, N_DIRS, SEED = 0.5, 200, 34
S_LO, S_HI = 0.1, 10.0          # rescale bounds from the design doc


def induced(model, X0, V, h=H):
    """|f(x + h*v) - f(x - h*v)| per point. V may be (d,) or (m,d)."""
    V = np.atleast_2d(V)
    if V.shape[0] == 1:
        V = np.repeat(V, len(X0), axis=0)
    return np.abs(model.predict(X0 + h * V) - model.predict(X0 - h * V))


def ds_test(model, X0, e, v_field, n_dirs=N_DIRS, seed=SEED):
    """
    Directional Specificity with a magnitude-matched null.
    Returns (DS, percentile, n_matched, check2_percentile).
    """
    rng = np.random.default_rng(seed)
    d = X0.shape[1]
    delta_v = induced(model, X0, v_field)
    ds_v = stats.spearmanr(delta_v, e).correlation
    target = float(np.mean(delta_v))

    dirs = rng.normal(size=(n_dirs, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    ds_null, raw_null, n_matched = [], [], 0
    for u in dirs:
        d_raw = induced(model, X0, u)
        raw_null.append(float(np.mean(d_raw)))
        m = float(np.mean(d_raw))
        if m <= 0:
            continue
        s = target / m
        if not (S_LO <= s <= S_HI):
            continue                      # unmatched, dropped per the design
        d_m = induced(model, X0, u, h=H * s)   # RECOMPUTED at the new step
        r = stats.spearmanr(d_m, e).correlation
        if np.isfinite(r):
            ds_null.append(r); n_matched += 1
    ds_null = np.array(ds_null)
    pct = float((ds_null < ds_v).mean()) if len(ds_null) else np.nan
    # Check 2 for comparison: raw magnitude vs raw null
    c2 = float((np.array(raw_null) < target).mean())
    return float(ds_v), pct, n_matched, c2


def make_case(kind, n=3000, d=8, seed=0, hyper=False):
    """
    kind='signal': difficulty varies along a known embedded direction w.
    kind='null'  : homoscedastic noise, no difficulty direction.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d); w[0] = 1.0          # the embedded difficulty direction
    f = 2.0 * X[:, 1] + 1.5 * X[:, 2] - X[:, 3]      # signal drives the MEAN
    if kind == "signal":
        sd = 0.3 + 2.0 * np.maximum(X @ w, 0)        # noise grows along w
    else:
        sd = np.full(n, 1.0)                          # constant noise
    y = f + rng.normal(0, sd)
    params = dict(n_estimators=400 if hyper else 200, learning_rate=0.06,
                  max_depth=8 if hyper else 4, verbose=-1, random_state=seed)
    tr, te = np.arange(0, n - 600), np.arange(n - 600, n)
    sc = StandardScaler().fit(X[tr])
    m = lgb.LGBMRegressor(**params).fit(sc.transform(X[tr]), y[tr])
    X0 = sc.transform(X[te])
    e = np.abs(m.predict(X0) - y[te])
    return m, X0, e, w


if __name__ == "__main__":
    print("=" * 96)
    print("TASK 34 Item 2 — synthetic verification of the DS test")
    print("=" * 96)
    rows = []
    for label, kind, hyper, must in [
            ("A_known_signal", "signal", False, "DETECT"),
            ("B_known_null", "null", False, "REJECT"),
            ("C_known_null_hyper", "null", True, "REJECT")]:
        m, X0, e, w = make_case(kind, hyper=hyper, seed=SEED)
        # candidate direction = the TRUE embedded direction for the signal case;
        # for null cases the same fixed direction (there is no true one)
        ds, pct, nm, c2 = ds_test(m, X0, e, w)
        resp = float(np.median(induced(m, X0, w)) / (np.std(e) + 1e-12))
        detected = bool(pct >= 0.95)
        ok = (detected if must == "DETECT" else not detected)
        print(f"\n  {label}  (must {must})")
        print(f"    responsiveness (median induced / err sd) = {resp:.3f}")
        print(f"    DS(candidate) = {ds:+.4f}   matched null dirs = {nm}/{N_DIRS}")
        print(f"    DS percentile = {pct:.3f}  -> {'DETECT' if detected else 'REJECT'}"
              f"   {'PASS' if ok else 'FAIL'}")
        print(f"    [Check 2 percentile on same data = {c2:.3f}]")
        rows.append(dict(case=label, requirement=must,
                         responsiveness=round(resp, 4),
                         DS=round(ds, 4), DS_percentile=round(pct, 4),
                         n_matched_dirs=nm, verdict=("DETECT" if detected else "REJECT"),
                         correct=ok, check2_percentile=round(c2, 4)))
    V = pd.DataFrame(rows)
    V.to_csv(f"{OUT}/task34_2_synthetic.csv", index=False)
    print("\n" + "=" * 96)
    print(V.to_string(index=False))
    print(f"\n  ALL CASES CORRECT: {bool(V.correct.all())}")
    print(f"\nSaved {OUT}/task34_2_synthetic.csv")
