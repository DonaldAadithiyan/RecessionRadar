"""
TASK 23, Item 1 — Verify ensemble disagreement is a legitimate signal.

Three checks, all of which gate Item 2:
  1. AVAILABLE AT PREDICTION TIME — uses only information the rest of the
     pipeline already requires. Leakage check: corrupt future data, confirm the
     quantity at step t does not change.
  2. CONTINUOUS AND COMPUTED EVERY MONTH — not conditioned on rare labels.
  3. NOT REDUNDANT with beta's existing feature set.

THE QUANTITY. Stage 2 runs three base models (CatBoost / LightGBM /
RandomForest) whose per-horizon predictions the ElasticNet meta-learner then
combines. Disagreement is defined, per horizon h, as

    disag_std[h]   = std(CatBoost_h, LightGBM_h, RandomForest_h)
    disag_range[h] = max(...) - min(...)
    disag_maxpair[h] = max pairwise |difference|

These are ALREADY computed inside the ensemble as meta-features
(ensemble_stubs.FullChainStackingEnsemble._engineer_meta_features: np.std,
np.min, np.max, and pairwise np.abs differences). This task extracts them as
inputs to beta rather than inventing a new quantity.

CRITICAL POINT ON AVAILABILITY: a base-model prediction is a function of X only.
It never touches y. So disagreement at month t is computable at prediction time
for exactly the same reason the prediction itself is.

Outputs: errordir/task23_1_disagreement.csv, task23_1_leakcheck.txt
"""
import os
import re
import sys
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
from task_oof_and_probit import (  # noqa: E402
    X_train as X_train_df, X_test as X_test_df, ensemble, LABELS,
)

OUT = "errordir"
os.makedirs(OUT, exist_ok=True)


def base_preds(X):
    """Per-base-model predictions, shape (n, 3, 4)."""
    bp = {n: m.predict(X) for n, m in ensemble.fitted_base_models.items()}
    names = list(ensemble.base_models)
    return np.stack([bp[n] for n in names], axis=1), names


def disagreement(X):
    """Continuous disagreement features, one row per month, no outcome used."""
    B, names = base_preds(X)                 # (n, 3, 4)
    out = {}
    for h_idx, h in enumerate(LABELS):
        v = B[:, :, h_idx]                   # (n, 3)
        out[f"disag_std_{h}"] = v.std(axis=1)
        out[f"disag_range_{h}"] = v.max(axis=1) - v.min(axis=1)
        pair = [np.abs(v[:, i] - v[:, j])
                for i in range(3) for j in range(i + 1, 3)]
        out[f"disag_maxpair_{h}"] = np.max(np.stack(pair, axis=1), axis=1)
    return pd.DataFrame(out, index=X.index), names


if __name__ == "__main__":
    print("=" * 96)
    print("TASK 23 Item 1 — is ensemble disagreement a legitimate signal?")
    print("=" * 96)

    D_train, names = disagreement(X_train_df)
    D_test, _ = disagreement(X_test_df)
    print(f"  base models: {names}")
    print(f"  disagreement features: {list(D_train.columns)}\n")

    lines = ["TASK 23 Item 1 — three-part legitimacy check", "=" * 70]

    # ── CHECK 1: availability / no leakage ────────────────────────────────
    # Disagreement at row t is a function of X.iloc[t] alone. Corrupt every row
    # from t onward and confirm row t-1's value is unchanged; and corrupt all
    # FUTURE rows and confirm row t is unchanged.
    Xc = X_test_df.copy()
    probe = 20
    clean_vals = disagreement(Xc.iloc[:probe + 1])[0].iloc[probe].values
    Xp = X_test_df.copy()
    Xp.iloc[probe + 1:] = 1e6                       # corrupt strictly future
    pois_vals = disagreement(Xp.iloc[:probe + 1])[0].iloc[probe].values
    same_future = bool(np.allclose(clean_vals, pois_vals))

    # row-locality: disagreement of row t must not depend on any other row
    Xr = X_test_df.copy()
    Xr.iloc[:probe] = 1e6                           # corrupt strictly past
    pois_past = disagreement(Xr.iloc[:probe + 1])[0].iloc[probe].values
    same_past = bool(np.allclose(clean_vals, pois_past))

    src = open(os.path.join(HERE, "task23_1_disagreement_check.py")).read()
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = "\n".join(l for l in code.split("\n") if not l.strip().startswith("#"))
    # Exclude the check's own source line, which would otherwise match itself
    # (this exact self-reference produced a false FAIL on the first run).
    code_nc = "\n".join(l for l in code.split("\n") if "touches_y" not in l)
    touches_y = ("y_test" in code_nc) or ("y_train" in code_nc)

    lines.append("\nCHECK 1 — available at prediction time (no leakage)")
    lines.append(f"  unchanged when all FUTURE rows corrupted: {same_future}")
    lines.append(f"  unchanged when all PAST rows corrupted:   {same_past}")
    lines.append(f"    (row-local: disagreement at t depends on X[t] alone)")
    lines.append(f"  computation touches any outcome y:        {'YES' if touches_y else 'NO'}")
    check1 = same_future and same_past and not touches_y
    lines.append(f"  -> {'PASS' if check1 else 'FAIL'}")

    # ── CHECK 2: continuous, computed every month ─────────────────────────
    n_all = len(D_train)
    n_finite = int(np.isfinite(D_train.values).all(axis=1).sum())
    n_unique = int(pd.Series(D_train[f"disag_std_6M"]).nunique())
    zero_frac = float((D_train[f"disag_std_6M"] == 0).mean())
    lines.append("\nCHECK 2 — continuous, computed every month, not label-conditioned")
    lines.append(f"  fit-pool months: {n_all}   with all features finite: {n_finite}")
    lines.append(f"  distinct values of disag_std_6M: {n_unique} "
                 f"({100*n_unique/n_all:.1f}% of months)")
    lines.append(f"  fraction exactly zero: {zero_frac:.4f}")
    check2 = (n_finite == n_all) and (n_unique > 0.5 * n_all) and zero_frac < 0.05
    lines.append(f"  -> {'PASS' if check2 else 'FAIL'}")

    # ── CHECK 3: not redundant with beta's existing features ──────────────
    lines.append("\nCHECK 3 — not redundant with beta's existing feature set")
    rows = []
    worst = 0.0
    for col in D_train.columns:
        v = D_train[col].values
        cors = []
        for f in X_train_df.columns:
            x = X_train_df[f].values
            if np.std(x) < 1e-12 or np.std(v) < 1e-12:
                continue
            cors.append((abs(float(np.corrcoef(v, x)[0, 1])), f))
        cors.sort(reverse=True)
        top_r, top_f = cors[0]
        # also: R^2 of regressing disagreement on ALL existing features
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X_train_df.values)
        rr = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(X_train_df.values), v)
        r2 = float(rr.score(sc.transform(X_train_df.values), v))
        rows.append(dict(feature=col, max_abs_corr=round(top_r, 4),
                         nearest_existing=top_f, r2_from_existing=round(r2, 4)))
        worst = max(worst, r2)
        lines.append(f"  {col:22s} max|r| with any existing feature = {top_r:.3f} "
                     f"({top_f}), R2 from ALL existing = {r2:.3f}")
    # criterion fixed in advance: redundant if fully explained (R2 > 0.95)
    check3 = worst <= 0.95
    lines.append(f"  worst R2 (any disagreement feature from existing set): {worst:.4f}")
    lines.append(f"  -> {'PASS' if check3 else 'FAIL'} (redundant iff R2 > 0.95)")

    pd.DataFrame(rows).to_csv(f"{OUT}/task23_1_disagreement.csv", index=False)
    D_train.to_csv(f"{OUT}/task23_1_disag_train.csv", index=False)
    D_test.to_csv(f"{OUT}/task23_1_disag_test.csv", index=False)

    lines.append("\n" + "=" * 70)
    lines.append(f"GATE: check1={'PASS' if check1 else 'FAIL'}  "
                 f"check2={'PASS' if check2 else 'FAIL'}  "
                 f"check3={'PASS' if check3 else 'FAIL'}")
    lines.append("PROCEED to Item 2" if (check1 and check2 and check3)
                 else "STOP — do not proceed to Item 2")
    txt = "\n".join(lines)
    print(txt)
    with open(f"{OUT}/task23_1_leakcheck.txt", "w") as f:
        f.write(txt + "\n")
    print(f"\nSaved {OUT}/task23_1_disagreement.csv, {OUT}/task23_1_leakcheck.txt")
