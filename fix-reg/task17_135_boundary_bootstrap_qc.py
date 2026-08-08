"""
TASK 17, Items 3 / 1 / 5 — boundary verification, bootstrap uncertainty, and the
Q_C-vs-support-width framing question.

Run in the spec's suggested order (3 verifies an existing proof, 1 backs the
paper's most-quoted number, 5 settles a wording question).

ITEM 3 — sweep the alpha boundary directly.
    Section 4.3 derives q >= 0.5 analytically from an oversupply argument: the
    alternating-tail rule always supplies ~N/2 extreme scores while the (1-alpha)
    quantile depends on only the top (1-q)*N of the set. This sweeps alpha from
    0.01 to 0.80 and measures the ceiling ratio directly.
    PRE-REGISTERED: ratio exactly 1.0 for every alpha <= 0.5, dropping below 1.0
    beyond it. Disagreement inside alpha <= 0.5 is a CRITICAL bug in the proof or
    its implementation and is reported as such, not smoothed over.

ITEM 1 — block bootstrap on the R^2 headline.
    The 0.85-vs-0.02 number carries no uncertainty. Bootstrapping the 200 draws
    directly would double-count their dependence on the pool, so the POOL is
    resampled (block length 12, matching the paper's autocorrelation choice) and
    the 200 draws regenerated inside each replicate. Full distributions are
    saved, not just endpoints.

ITEM 5 — does support width add anything once Q_C is in the model?
    The reverse of Task 16B. If support width's partial R^2 given Q_C is
    negligible, Q_C is the operative quantity and support width is a diagnostic
    proxy; the paper's Section 4 should lead with Q_C. Determined by the number,
    not by preference.

Outputs:
  fix-reg/task17_3_alpha_boundary.csv
  fix-reg/task17_1_bootstrap.csv
  fix-reg/task17_1_bootstrap_summary.csv
  fix-reg/task17_5_qc_vs_support.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()

from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, random_draw_sweep, GAMMA_DEFAULT,
)
import selector_lib as SEL  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
N_DRAWS = 200
SEED = 7
B_BOOT = 1000
BLOCK = 12
LABELS = ["Current", "1M", "3M", "6M"]

ALPHA_GRID = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
              0.49, 0.50, 0.55, 0.60, 0.70, 0.80]

print("=" * 96)
print("TASK 17 — Items 3, 1, 5")
print("=" * 96)

import task_oof_and_probit as T  # noqa: E402
IS_RARE = (T.train_df["recession_probability"].values >= 50)


def scores_for(h_idx):
    return np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx])


# =============================================================================
# ITEM 3 — alpha boundary sweep
# =============================================================================
print("\n" + "-" * 96)
print("ITEM 3 — ceiling ratio vs alpha (pre-registered: 1.0 for alpha <= 0.5)")
print("-" * 96)

i3_rows = []
for h_idx, h in enumerate(LABELS):
    s = scores_for(h_idx)
    p = s[np.isfinite(s)]
    if len(p) < N_FIX + 5:
        continue
    order = np.argsort(p)
    top = p[order[-N_FIX:]]                    # ceiling-attaining subset
    sel = p[SEL.support_width_selector(p, N_FIX)]

    for a in ALPHA_GRID:
        q = 1.0 - a
        c_sel = float(np.quantile(sel, q))
        c_top = float(np.quantile(top, q))
        ratio = c_sel / c_top if c_top > 0 else np.nan
        i3_rows.append(dict(horizon=h, alpha=a, q=round(q, 3), N=N_FIX,
                            selector_q=round(c_sel, 5),
                            ceiling_q=round(c_top, 5),
                            ratio=round(ratio, 6) if np.isfinite(ratio) else None,
                            within_proven_region=bool(a <= 0.5),
                            at_ceiling=bool(np.isfinite(ratio) and
                                            ratio >= 1 - 1e-9)))

I3 = pd.DataFrame(i3_rows)
I3.to_csv(f"{OUT}/task17_3_alpha_boundary.csv", index=False)

piv3 = I3.pivot_table(index="alpha", columns="horizon", values="ratio")
print(piv3.round(4).to_string())

proven = I3[I3["within_proven_region"]]
violations = proven[~proven["at_ceiling"]]
print(f"\n  alpha <= 0.5 configurations: {len(proven)}, "
      f"at ceiling: {int(proven['at_ceiling'].sum())}")
if len(violations):
    print("  *** CRITICAL: the analytical boundary is VIOLATED inside the "
          "proven region ***")
    print(violations[["horizon", "alpha", "ratio"]].to_string(index=False))
else:
    print("  -> proof VERIFIED empirically: every alpha <= 0.5 sits exactly "
          "at the ceiling")
beyond = I3[~I3["within_proven_region"]]
print(f"  alpha > 0.5: {len(beyond)} configs, "
      f"below ceiling in {int((~beyond['at_ceiling']).sum())} "
      f"(min ratio {beyond['ratio'].min():.4f})")


# =============================================================================
# ITEM 1 — block bootstrap on the R^2 headline
# =============================================================================
print("\n" + "-" * 96)
print(f"ITEM 1 — block bootstrap (B={B_BOOT}, block={BLOCK} months) on R^2")
print("-" * 96)
print("  Resampling the POOL, not the draws: the 200 draws are derived from the")
print("  pool, so bootstrapping them directly would double-count that dependency.")


def block_resample(n, block, rng):
    """Moving-block resample of indices 0..n-1, preserving autocorrelation."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


boot_rows = []
for h_idx, h in [(2, "3M"), (3, "6M")]:
    s_all = scores_for(h_idx)
    y_te = T.y_test[:, h_idx]
    p_te = T.preds_test[:, h_idx]
    rng = np.random.default_rng(SEED)
    n_pool = len(s_all)

    print(f"\n  {h}: running {B_BOOT} replicates...")
    for b in range(B_BOOT):
        idx = block_resample(n_pool, BLOCK, rng)
        s_b = s_all[idx]
        rare_b = IS_RARE[idx]
        valid = np.isfinite(s_b)
        if valid.sum() < N_FIX + 10:
            continue
        sw = random_draw_sweep(s_b, rare_b, y_te, p_te, N=N_FIX,
                               n_draws=N_DRAWS, seed=int(rng.integers(1e9)))
        cov = sw["cov"].values
        if np.std(cov) < 1e-12:
            continue
        supp = sw["supp"].values
        rare = sw["rare_count"].values.astype(float)
        r2_s = (st.pearsonr(supp, cov).statistic ** 2
                if np.std(supp) > 0 else np.nan)
        r2_r = (st.pearsonr(rare, cov).statistic ** 2
                if np.std(rare) > 0 else np.nan)
        boot_rows.append(dict(horizon=h, replicate=b,
                              R2_supp=r2_s, R2_rare=r2_r,
                              delta_R2=r2_s - r2_r))

BOOT = pd.DataFrame(boot_rows)
BOOT.to_csv(f"{OUT}/task17_1_bootstrap.csv", index=False)

summ = []
for h in ["3M", "6M"]:
    d = BOOT[BOOT["horizon"] == h]
    if d.empty:
        continue
    row = dict(horizon=h, B_effective=len(d))
    for col in ["R2_supp", "R2_rare", "delta_R2"]:
        v = d[col].dropna().values
        row[f"{col}_median"] = round(float(np.median(v)), 4)
        row[f"{col}_lo95"] = round(float(np.percentile(v, 2.5)), 4)
        row[f"{col}_hi95"] = round(float(np.percentile(v, 97.5)), 4)
    row["delta_excludes_zero"] = bool(row["delta_R2_lo95"] > 0)
    summ.append(row)
    print(f"\n  {h}  (B_effective={len(d)})")
    print(f"    R2(support)  median={row['R2_supp_median']:.4f}  "
          f"95% [{row['R2_supp_lo95']:.4f}, {row['R2_supp_hi95']:.4f}]")
    print(f"    R2(rare)     median={row['R2_rare_median']:.4f}  "
          f"95% [{row['R2_rare_lo95']:.4f}, {row['R2_rare_hi95']:.4f}]")
    print(f"    delta R2     median={row['delta_R2_median']:.4f}  "
          f"95% [{row['delta_R2_lo95']:.4f}, {row['delta_R2_hi95']:.4f}]  "
          f"excludes 0: {row['delta_excludes_zero']}")

S1 = pd.DataFrame(summ)
S1.to_csv(f"{OUT}/task17_1_bootstrap_summary.csv", index=False)


# =============================================================================
# ITEM 5 — does support width add anything once Q_C is in the model?
# =============================================================================
print("\n" + "-" * 96)
print("ITEM 5 — partial R^2 of support width GIVEN Q_C (reverse of Task 16B)")
print("-" * 96)


def standardize(v):
    v = np.asarray(v, float)
    sd = np.std(v)
    return (v - v.mean()) / sd if sd > 1e-12 else v * 0.0


i5_rows = []
for h_idx, h in [(2, "3M"), (3, "6M")]:
    s_all = scores_for(h_idx)
    sw = random_draw_sweep(s_all, IS_RARE, T.y_test[:, h_idx],
                           T.preds_test[:, h_idx], N=N_FIX,
                           n_draws=N_DRAWS, seed=SEED)
    y = standardize(sw["cov"].values)
    qc = standardize(sw["Q_C"].values)
    supp = standardize(sw["supp"].values)

    def fit(cols):
        X = np.column_stack([np.ones(len(y))] + cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return beta, float(resid @ resid)

    b_full, ssr_full = fit([qc, supp])
    _, ssr_qc_only = fit([qc])
    _, ssr_supp_only = fit([supp])
    sst = float(((y - y.mean()) ** 2).sum())

    # partial R^2 of support width given Q_C already in the model
    pr2_supp_given_qc = ((ssr_qc_only - ssr_full) / ssr_qc_only
                         if ssr_qc_only > 0 else 0.0)
    # and the mirror image, for context
    pr2_qc_given_supp = ((ssr_supp_only - ssr_full) / ssr_supp_only
                         if ssr_supp_only > 0 else 0.0)

    n, k = len(y), 3
    sigma2 = ssr_full / (n - k)
    X = np.column_stack([np.ones(n), qc, supp])
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sigma2)
    tcrit = st.t.ppf(0.975, n - k)

    i5_rows.append(dict(
        horizon=h, n_draws=n,
        beta_QC=round(float(b_full[1]), 4),
        beta_QC_lo=round(float(b_full[1] - tcrit * se[1]), 4),
        beta_QC_hi=round(float(b_full[1] + tcrit * se[1]), 4),
        beta_supp=round(float(b_full[2]), 4),
        beta_supp_lo=round(float(b_full[2] - tcrit * se[2]), 4),
        beta_supp_hi=round(float(b_full[2] + tcrit * se[2]), 4),
        partial_R2_supp_given_QC=round(pr2_supp_given_qc, 4),
        partial_R2_QC_given_supp=round(pr2_qc_given_supp, 4),
        R2_QC_only=round(1 - ssr_qc_only / sst, 4),
        R2_supp_only=round(1 - ssr_supp_only / sst, 4),
        R2_both=round(1 - ssr_full / sst, 4),
        supp_ci_excludes_zero=bool(
            (b_full[2] - tcrit * se[2]) * (b_full[2] + tcrit * se[2]) > 0)))
    r = i5_rows[-1]
    print(f"\n  {h}:")
    print(f"    R2: Q_C alone={r['R2_QC_only']:.4f}  "
          f"support alone={r['R2_supp_only']:.4f}  both={r['R2_both']:.4f}")
    print(f"    beta(Q_C)={r['beta_QC']:+.3f} "
          f"[{r['beta_QC_lo']:+.3f},{r['beta_QC_hi']:+.3f}]")
    print(f"    beta(supp)={r['beta_supp']:+.3f} "
          f"[{r['beta_supp_lo']:+.3f},{r['beta_supp_hi']:+.3f}]  "
          f"CI excludes 0: {r['supp_ci_excludes_zero']}")
    print(f"    partial R2 of SUPPORT given Q_C = "
          f"{r['partial_R2_supp_given_QC']:.4f}")
    print(f"    partial R2 of Q_C given SUPPORT = "
          f"{r['partial_R2_QC_given_supp']:.4f}")

I5 = pd.DataFrame(i5_rows)
I5.to_csv(f"{OUT}/task17_5_qc_vs_support.csv", index=False)

print("\n" + "=" * 96)
print("SUMMARY")
print("=" * 96)
print(f"  Item 3: proof {'VERIFIED' if not len(violations) else 'VIOLATED'} "
      f"for alpha <= 0.5")
print(f"  Item 1: delta R^2 95% interval excludes zero at "
      f"{int(S1['delta_excludes_zero'].sum())} of {len(S1)} horizons")
print(f"  Item 5: partial R^2 of support given Q_C = "
      + ", ".join(f"{r['horizon']}:{r['partial_R2_supp_given_QC']:.4f}"
                  for r in i5_rows))
print(f"\nSaved task17_3_alpha_boundary.csv, task17_1_bootstrap.csv, "
      f"task17_5_qc_vs_support.csv")
