"""
TASK 9 PART B — Tune Bellman Conformal Inference properly.

BCI was previously reported only with the authors' untuned defaults
(lambda_init=5, lambda_max=500, gamma=0.8, T=3). Task 7 already established
that a hand-set lambda can move BCI's coverage anywhere from 3% to 94%, so the
untuned number is not a ceiling on what BCI can do. This gives it a fair,
documented tuning pass.

TUNING PROTOCOL (causal-window discipline — test data is never touched):
  - The validation slice is carved from the CALIBRATION POOL only. For each
    domain the pool is split temporally: the first 80% supplies calibration
    scores, the final 20% acts as a held-out validation stream on which BCI's
    parameters are scored. The real test window plays no part in selection.
  - Grid search over lambda_init, lambda_max and gamma (the latter spanning
    0.5x to 4x the authors' default, per the task spec).
  - Objective: minimise |coverage - 90%| on the validation stream. Ties broken
    toward the narrower mean interval, so the search cannot buy coverage with
    vacuous width.
  - The selected parameters are then applied, unchanged, to the same
    out-of-fold test evaluation every other strategy receives.

Tuned BCI is reported ALONGSIDE untuned, never replacing it: the gap between
them is itself the finding about BCI's parameter sensitivity.

Outputs:
  fix-reg/task9b_bci_tuning.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()

from domain_common import coverage_and_width, wilson_ci_from_indicator, \
    block_bootstrap_ci  # noqa: E402
import baselines_lib as B  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
ALPHA_TARGET = 0.10

# Grid: centred on the authors' released defaults, widened per the task spec.
GRID_LAMBDA_INIT = [1.0, 5.0, 20.0, 50.0]
GRID_LAMBDA_MAX = [50.0, 200.0, 500.0, 2000.0]
GRID_GAMMA = [0.4, 0.8, 1.6, 3.2]          # 0.5x .. 4x the default 0.8
DEFAULTS = dict(lambda_init=5.0, lambda_max=500.0, gamma=0.8)


def tune_bci(cal_scores, val_y, val_p, target=0.90):
    """
    Grid-search BCI parameters on a held-out validation stream.
    Returns (best_params, best_val_coverage, best_val_width, n_evaluated).
    """
    best, best_key = None, None
    n_eval = 0
    for li in GRID_LAMBDA_INIT:
        for lm in GRID_LAMBDA_MAX:
            if lm <= li:
                continue
            for g in GRID_GAMMA:
                cov_arr, w_arr = B.run_bci(val_y, val_p, cal_scores,
                                           lambda_init=li, lambda_max=lm,
                                           gamma=g)
                cov, w = coverage_and_width(cov_arr, w_arr)
                n_eval += 1
                if not np.isfinite(cov):
                    continue
                # primary: distance from nominal; tiebreak: narrower interval
                key = (round(abs(cov - target * 100), 4), w)
                if best_key is None or key < best_key:
                    best_key = key
                    best = dict(lambda_init=li, lambda_max=lm, gamma=g,
                                val_cov=round(cov, 2), val_width=round(w, 3))
    return best, n_eval


def evaluate(name, domain, model, horizon, scores_pool, y_te, p_te, block,
             params, tag):
    s = np.asarray(scores_pool, dtype=float)
    valid = np.where(np.isfinite(s))[0]
    trailing = s[valid[-N_FIX:]] if len(valid) >= N_FIX else s[valid]
    cov_arr, w_arr = B.run_bci(y_te, p_te, trailing, **params)
    cov, w = coverage_and_width(cov_arr, w_arr)
    wlo, whi = wilson_ci_from_indicator(cov_arr)
    blo, bhi = block_bootstrap_ci(cov_arr, block=block)
    yv = np.asarray(y_te, float); yv = yv[np.isfinite(yv)]
    spread = float(np.percentile(yv, 95) - np.percentile(yv, 5))
    return dict(domain=domain, model=model, horizon=horizon, variant=tag,
                n=int(np.isfinite(np.asarray(cov_arr, float)).sum()),
                coverage=round(cov, 2), wilson_lo=round(wlo, 2),
                wilson_hi=round(whi, 2), boot_lo=round(blo, 2),
                boot_hi=round(bhi, 2), mean_width=round(w, 3),
                width_ratio=round(w / spread, 1) if spread > 0 else None,
                **{f"p_{k}": v for k, v in params.items()})


print("=" * 78)
print("TASK 9B — Bellman Conformal Inference: tuned vs untuned")
print("=" * 78)
print(f"  Grid: lambda_init {GRID_LAMBDA_INIT}")
print(f"        lambda_max  {GRID_LAMBDA_MAX}")
print(f"        gamma       {GRID_GAMMA}  (0.5x-4x the authors' 0.8)")
print(f"  Authors' defaults: {DEFAULTS}")

rows = []


def do_domain(domain, model, horizon, scores_pool, y_te, p_te, block):
    """Split the pool into calibration + validation, tune, then evaluate."""
    s = np.asarray(scores_pool, dtype=float)
    valid = np.where(np.isfinite(s))[0]
    if len(valid) < 60:
        print(f"  [skip] {domain}/{model}/{horizon}: too few scored points")
        return

    # Temporal split of the POOL only. The last 20% of scored pool points
    # becomes the validation stream; calibration comes from the first 80%.
    cut = int(0.8 * len(valid))
    cal_idx, val_idx = valid[:cut], valid[cut:]
    cal_scores = s[cal_idx][-N_FIX:]

    # The validation stream needs (y, prediction) pairs. Pool points are
    # scored by |pred - y|, so a synthetic stream is reconstructed with the
    # pool's own errors around a zero-centred prediction: this preserves the
    # error distribution BCI must adapt to without using any test data.
    val_err = s[val_idx]
    val_y = val_err.astype(float)
    val_p = np.zeros_like(val_y)

    best, n_eval = tune_bci(cal_scores, val_y, val_p)
    if best is None:
        print(f"  [skip] {domain}/{model}/{horizon}: no valid grid point")
        return
    tuned = dict(lambda_init=best["lambda_init"],
                 lambda_max=best["lambda_max"], gamma=best["gamma"])
    print(f"\n  {domain}/{model}/{horizon}: searched {n_eval} configs on "
          f"{len(val_idx)} validation points")
    print(f"    selected {tuned}  (val coverage {best['val_cov']}%, "
          f"val width {best['val_width']})")

    rows.append(evaluate("bci", domain, model, horizon, scores_pool,
                         y_te, p_te, block, DEFAULTS, "untuned"))
    rows.append(evaluate("bci", domain, model, horizon, scores_pool,
                         y_te, p_te, block, tuned, "tuned"))
    a, b = rows[-2], rows[-1]
    print(f"    TEST untuned={a['coverage']:6.2f}% (w={a['mean_width']:.2f})  "
          f"tuned={b['coverage']:6.2f}% (w={b['mean_width']:.2f})")


# ── Recession ────────────────────────────────────────────────────────────────
print("\n" + "-" * 78)
print("RECESSION")
print("-" * 78)
import task_oof_and_probit as T  # noqa: E402
LABELS = ["Current", "1M", "3M", "6M"]
for h_idx, h in enumerate(LABELS):
    s_oof = np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx])
    do_domain("Recession", "stacking-chain", h, s_oof,
              T.y_test[:, h_idx], T.preds_test[:, h_idx], block=12)

# ── Healthcare (internal models + Task 8 literature baselines) ──────────────
print("\n" + "-" * 78)
print("HEALTHCARE")
print("-" * 78)
import runpy  # noqa: E402
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    do_domain("Healthcare", name, "30-day", g["SCORES"][name],
              g["Y_TEST"], g["PREDS_TEST"][name], block=1)

# ── Climate ─────────────────────────────────────────────────────────────────
print("\n" + "-" * 78)
print("CLIMATE")
print("-" * 78)
gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    do_domain("Climate", name, "region-month", gc["SCORES"][name],
              gc["Y_TEST"], gc["PREDS_TEST"][name], block=12)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/task9b_bci_tuning.csv", index=False)

print("\n" + "=" * 78)
print("TUNED vs UNTUNED — all domains")
print("=" * 78)
piv = res.pivot_table(index=["domain", "model", "horizon"], columns="variant",
                      values=["coverage", "mean_width"])
print(piv.round(2).to_string())

print("\n" + "=" * 78)
print("DOES TUNED BCI BEAT THE DIVERSITY-OPTIMAL SELECTOR ANYWHERE?")
print("=" * 78)
comp = []
for f, dom in [("task7_baselines_recession.csv", "Recession"),
               ("task7_baselines_healthcare.csv", "Healthcare"),
               ("task7_baselines_climate.csv", "Climate")]:
    p = f"{OUT}/{f}"
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    if "scoring" in d and (d.scoring == "out-of-fold").any():
        d = d[d.scoring == "out-of-fold"]
    for _, r in d[d.strategy == "diversity_optimal"].iterrows():
        t = res[(res.domain == dom) & (res.model == r.model) &
                (res.horizon == r.horizon) & (res.variant == "tuned")]
        if t.empty:
            continue
        t = t.iloc[0]
        comp.append(dict(domain=dom, model=r.model, horizon=r.horizon,
                         divopt=r.coverage, bci_tuned=t.coverage,
                         divopt_w=r.mean_width, bci_w=t.mean_width,
                         bci_beats=bool(t.wilson_lo > r.coverage)))
if comp:
    c = pd.DataFrame(comp)
    print(c.to_string(index=False))
    print(f"\n  Cases where tuned BCI separates above diversity-optimal: "
          f"{int(c['bci_beats'].sum())} of {len(c)}")
print(f"\nSaved {OUT}/task9b_bci_tuning.csv")
