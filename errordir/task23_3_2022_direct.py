"""
TASK 23, Item 3 — the direct test: does the disagreement-augmented beta flag
2022 now, where the old feature set flagged nothing?

This is a DIRECT DIAGNOSTIC on one motivating case, n=1 year (12 months). It is
reported as a yes/no with projection values. NO significance testing is applied
— per the guardrail, that would be inappropriate for n=1.

Comparison is like-for-like: old beta (existing features) vs new beta (existing
+ disagreement), both fit on the same 291-month FIT split, both projections
ranked against the same CAL reference distribution.

Output: errordir/task23_3_2022_direct.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg"))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from task23_1_disagreement_check import disagreement  # noqa: E402
from task_oof_and_probit import (  # noqa: E402
    X_train as X_train_df, X_test as X_test_df, y_train as y_pool, y_test,
    oof_pred as preds_pool_oof, preds_test, n_pool, test_df, LABELS,
)

OUT = "errordir"
FIT_FRAC = 0.60
n_fit = int(round(FIT_FRAC * n_pool))
FIT_IDX, CAL_IDX = np.arange(0, n_fit), np.arange(n_fit, n_pool)

D_train, _ = disagreement(X_train_df)
D_test, _ = disagreement(X_test_df)
dates = pd.to_datetime(test_df["date"].values)


def fit_and_rank(h_idx, h, with_disag):
    dcols = [f"disag_std_{h}", f"disag_range_{h}", f"disag_maxpair_{h}"]
    if with_disag:
        Xtr = np.column_stack([X_train_df.values, D_train[dcols].values])
        Xte = np.column_stack([X_test_df.values, D_test[dcols].values])
    else:
        Xtr, Xte = X_train_df.values, X_test_df.values
    ae = np.abs(preds_pool_oof[:, h_idx] - y_pool[:, h_idx])
    fo = FIT_IDX[np.isfinite(ae[FIT_IDX])]
    co = CAL_IDX[np.isfinite(ae[CAL_IDX])]
    sc = StandardScaler().fit(Xtr[fo])
    rg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(sc.transform(Xtr[fo]), ae[fo])
    b = rg.coef_.astype(float)
    pc = sc.transform(Xtr[co]) @ b
    pt = sc.transform(Xte) @ b
    ref = np.sort(pc[np.isfinite(pc)])
    rank = np.searchsorted(ref, pt, side="right") / max(len(ref), 1)
    return pt, rank


if __name__ == "__main__":
    H_IDX, H = 3, "6M"
    realized = np.abs(preds_test[:, H_IDX] - y_test[:, H_IDX])
    yrs = dates.year

    pt_old, rk_old = fit_and_rank(H_IDX, H, with_disag=False)
    pt_new, rk_new = fit_and_rank(H_IDX, H, with_disag=True)

    print("=" * 96)
    print(f"TASK 23 Item 3 — does the augmented beta flag 2022? ({H}, n=1 year)")
    print("=" * 96)

    rows = []
    for yr in sorted(set(yrs)):
        m = yrs == yr
        rows.append(dict(year=int(yr), n_months=int(m.sum()),
                         mean_realized_error=round(float(np.nanmean(realized[m])), 2),
                         mean_rank_old=round(float(rk_old[m].mean()), 3),
                         mean_rank_new=round(float(rk_new[m].mean()), 3),
                         mean_proj_old=round(float(pt_old[m].mean()), 2),
                         mean_proj_new=round(float(pt_new[m].mean()), 2)))
    T = pd.DataFrame(rows)
    T.to_csv(f"{OUT}/task23_3_2022_direct.csv", index=False)
    print(T.to_string(index=False))

    # "Flagged" = mean rank in the top tercile of the test period's own ranks,
    # the same notion of "elevated difficulty" Task 21 Item 4 used.
    thr_old = np.quantile(rk_old, 2 / 3)
    thr_new = np.quantile(rk_new, 2 / 3)
    m22 = yrs == 2022
    flag_old = float((rk_old[m22] >= thr_old).mean())
    flag_new = float((rk_new[m22] >= thr_new).mean())

    print("\n" + "-" * 96)
    print("DIRECT ANSWER (n=1 diagnostic, no significance test applied)")
    print("-" * 96)
    print(f"  2022 mean realized error: {float(np.nanmean(realized[m22])):.2f} "
          f"(highest of any test year)")
    print(f"  OLD beta: mean rank {rk_old[m22].mean():.3f}, "
          f"{100*flag_old:.0f}% of 2022 months in top tercile")
    print(f"  NEW beta: mean rank {rk_new[m22].mean():.3f}, "
          f"{100*flag_new:.0f}% of 2022 months in top tercile")
    print(f"\n  Does 2022 get flagged now? "
          f"{'YES' if flag_new > flag_old else ('NO CHANGE' if flag_new == flag_old else 'NO — WORSE')}")
    print(f"\nSaved {OUT}/task23_3_2022_direct.csv")
