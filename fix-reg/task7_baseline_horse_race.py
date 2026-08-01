"""
TASK 7 — Cross-domain baseline comparison.

Runs the same set of calibration strategies against all three domains
(recession, healthcare, climate) so the standard-toolbox comparison is no
longer primary-testbed-only.

Strategies: pooled/trailing ACI, Mondrian, PID-conformal, EVT-tail,
diversity-optimal (ours), DtACI (Gibbs & Candes 2024), AcMCP (Wang & Hyndman
2024). Bellman CI and CPTC are addressed separately — see the write-up.

Order of work follows the spec: the recession rows are computed FIRST and
checked against the published Phase 3 figures. If they do not reproduce, that
is a bug and the run aborts rather than reporting new numbers on a broken base.

Outputs:
  fix-reg/task7_baselines_recession.csv
  fix-reg/task7_baselines_healthcare.csv
  fix-reg/task7_baselines_climate.csv
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
# The saved ensemble is pickled against __main__; register its classes here so
# that loading it works from this entry point too.
ensemble_stubs.install()

from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, support_width, wilson_ci_from_indicator,
    block_bootstrap_ci, GAMMA_DEFAULT,
)
import baselines_lib as B  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
LABELS = ["Current", "1M", "3M", "6M"]

# Phase 3's published recession figures — the reproduction target.
PHASE3_REF = {
    "pooled":   {"Current": 89.23, "1M": 85.94, "3M": 85.48, "6M": 67.80},
    "mondrian": {"Current": 81.54, "1M": 85.94, "3M": 85.48, "6M": 61.02},
    "pid":      {"Current": 89.23, "1M": 87.50, "3M": 87.10, "6M": 71.19},
    "evt":      {"Current": 87.69, "1M": 85.94, "3M": 85.48, "6M": 66.10},
}


def greedy_extreme(s_all, N=N_FIX):
    """Diversity-maximizing selector (ours) — same rule as task_phase2.py."""
    valid = np.where(~np.isnan(s_all))[0]
    order = valid[np.argsort(s_all[valid])]
    chosen, lo, hi, take_low = [], 0, len(order) - 1, True
    while len(chosen) < min(N, len(order)) and lo <= hi:
        if take_low:
            chosen.append(order[lo]); lo += 1
        else:
            chosen.append(order[hi]); hi -= 1
        take_low = not take_low
    return np.array(sorted(chosen))


def summarize(domain, model, horizon, scoring, strategy, covered, widths,
              y_te, block):
    """One output row, always with coverage + interval + width."""
    cov, w = coverage_and_width(covered, widths)
    wlo, whi = wilson_ci_from_indicator(covered)
    blo, bhi = block_bootstrap_ci(covered, block=block)
    yv = np.asarray(y_te, dtype=float)
    yv = yv[np.isfinite(yv)]
    spread = float(np.percentile(yv, 95) - np.percentile(yv, 5)) if len(yv) >= 2 else np.nan
    n = int(np.isfinite(np.asarray(covered, dtype=float)).sum())
    return dict(domain=domain, model=model, horizon=horizon, scoring=scoring,
                strategy=strategy, n=n,
                coverage=round(cov, 2),
                wilson_lo=round(wlo, 2), wilson_hi=round(whi, 2),
                boot_lo=round(blo, 2), boot_hi=round(bhi, 2),
                mean_width=round(w, 3),
                target_spread=round(spread, 3),
                width_ratio=round(w / spread, 1) if spread and spread > 0 else None,
                vacuous=bool(spread and spread > 0 and w > 100.0))


def run_all_strategies(domain, model, horizon, scoring, scores_pool, is_rare,
                       y_te, p_te, test_regime, block, horizon_steps=1):
    """Apply every strategy to one (domain, model, horizon) slice."""
    rows = []
    s = np.asarray(scores_pool, dtype=float)
    valid = np.where(np.isfinite(s))[0]
    trailing = s[valid[-N_FIX:]] if len(valid) >= N_FIX else s[valid]

    def add(strategy, covered, widths):
        rows.append(summarize(domain, model, horizon, scoring, strategy,
                              covered, widths, y_te, block))

    # 1. pooled / trailing ACI (reference row)
    cov, _, wid = run_aci(y_te, p_te, trailing, gamma=GAMMA_DEFAULT)
    add("pooled_trailing", cov, wid)

    # 2. Mondrian — class-conditional on the regime label
    cal_by_regime = {}
    reg_pool = is_rare[valid[-N_FIX:]] if len(valid) >= N_FIX else is_rare[valid]
    sc_pool = trailing
    for lab in np.unique(reg_pool):
        cal_by_regime[bool(lab)] = sc_pool[reg_pool == lab]
    cov, wid = B.run_mondrian(y_te, p_te, cal_by_regime,
                              [bool(x) for x in test_regime])
    add("mondrian", cov, wid)

    # 3. PID-conformal
    cov, wid = B.run_pid(y_te, p_te, trailing)
    add("pid_conformal", cov, wid)

    # 4. EVT-tail
    cov, wid = B.run_evt(y_te, p_te, trailing)
    add("evt_tail", cov, wid)

    # 5. diversity-optimal (ours)
    sel = greedy_extreme(s)
    cov, _, wid = run_aci(y_te, p_te, s[sel][np.isfinite(s[sel])],
                          gamma=GAMMA_DEFAULT)
    add("diversity_optimal", cov, wid)

    # 6. DtACI (Gibbs & Candes 2024)
    cov, wid = B.run_dtaci(y_te, p_te, trailing)
    add("dtaci", cov, wid)

    # 7. AcMCP (Wang & Hyndman 2024)
    cov, wid = B.run_acmcp(y_te, p_te, trailing, horizon=horizon_steps)
    add("acmcp", cov, wid)

    # 8. Bellman Conformal Inference (Yang, Candes & Lei 2024)
    cov, wid = B.run_bci(y_te, p_te, trailing)
    add("bellman_ci", cov, wid)

    return rows


# =============================================================================
# DOMAIN 1 — recession (in-sample, to reproduce Phase 3; and out-of-fold)
# =============================================================================
def domain_recession():
    from task_oof_and_probit import (  # reuses the identical setup
        y_train, y_test, preds_train_insample, preds_test, oof_pred,
        train_df, test_df, n_pool,
    )
    rows = []
    rec_prob_tr = train_df["recession_probability"].values
    te_regime = test_df["recession_probability"].values >= 50
    is_rare = rec_prob_tr >= 50
    horizon_steps = {"Current": 1, "1M": 1, "3M": 3, "6M": 6}

    for scoring, preds_pool in [("in-sample", preds_train_insample),
                                ("out-of-fold", oof_pred)]:
        for h_idx, h in enumerate(LABELS):
            s = np.abs(preds_pool[:, h_idx] - y_train[:, h_idx])
            rows += run_all_strategies(
                "Recession", "stacking-chain", h, scoring, s, is_rare,
                y_test[:, h_idx], preds_test[:, h_idx], te_regime,
                block=12, horizon_steps=horizon_steps[h])
    return pd.DataFrame(rows)


def check_phase3(df):
    """Abort if the in-sample recession rows do not reproduce Phase 3."""
    print("\n" + "=" * 78)
    print("PHASE 3 REPRODUCTION CHECK (in-sample recession rows)")
    print("=" * 78)
    ins = df[(df["scoring"] == "in-sample")]
    name_map = {"pooled": "pooled_trailing", "mondrian": "mondrian",
                "pid": "pid_conformal", "evt": "evt_tail"}
    ok = True
    for key, strat in name_map.items():
        for h in LABELS:
            want = PHASE3_REF[key][h]
            got_rows = ins[(ins["strategy"] == strat) & (ins["horizon"] == h)]
            if got_rows.empty:
                print(f"  MISSING {strat} {h}"); ok = False; continue
            got = float(got_rows["coverage"].iloc[0])
            delta = abs(got - want)
            flag = "OK " if delta < 0.01 else "MISMATCH"
            if delta >= 0.01:
                ok = False
            print(f"  {flag} {strat:18s} {h:8s} published={want:6.2f} "
                  f"recomputed={got:6.2f}  delta={delta:.2f}")
    return ok


# =============================================================================
# DOMAIN 2 — healthcare
# =============================================================================
def domain_healthcare():
    import runpy
    print("\nRebuilding healthcare domain state...")
    g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
    rows = []
    for name in ["ridge", "gradboost"]:
        s = g["SCORES"][name]
        p_te = g["PREDS_TEST"][name]
        y_te = g["Y_TEST"]
        is_rare = g["RARE_POOL"]
        # Mondrian regime for healthcare: top-tercile PREDICTED risk vs rest.
        # Documented as a design choice in the write-up, not a given.
        thr = np.percentile(p_te, 100 * 2 / 3)
        te_regime = p_te >= thr
        rows += run_all_strategies(
            "Healthcare", name, "30-day", "out-of-fold", s, is_rare,
            y_te, p_te, te_regime, block=1, horizon_steps=1)
    return pd.DataFrame(rows)


# =============================================================================
# DOMAIN 3 — climate
# =============================================================================
def domain_climate():
    import runpy
    print("\nRebuilding climate domain state...")
    g = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
    rows = []
    for name in ["ridge", "gradboost"]:
        s = g["SCORES"][name]
        p_te = g["PREDS_TEST"][name]
        y_te = g["Y_TEST"]
        is_rare = g["RARE_POOL"]
        te_month = g["TEST_MONTH"]
        # Mondrian regime for climate: named-storm-season (Jun-Nov) vs off.
        te_regime = np.isin(te_month, [6, 7, 8, 9, 10, 11])
        rows += run_all_strategies(
            "Climate", name, "region-month", "out-of-fold", s, is_rare,
            y_te, p_te, te_regime, block=12, horizon_steps=1)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 78)
    print("TASK 7 — cross-domain baseline horse race")
    print("=" * 78)

    rec = domain_recession()
    if not check_phase3(rec):
        print("\nPhase 3 reproduction FAILED — aborting before reporting new "
              "numbers, per the task spec's bug-check requirement.")
        rec.to_csv(f"{OUT}/task7_baselines_recession_DEBUG.csv", index=False)
        sys.exit(1)
    print("\n  Phase 3 reproduces exactly. Proceeding.")
    rec.to_csv(f"{OUT}/task7_baselines_recession.csv", index=False)

    hc = domain_healthcare()
    hc.to_csv(f"{OUT}/task7_baselines_healthcare.csv", index=False)

    cl = domain_climate()
    cl.to_csv(f"{OUT}/task7_baselines_climate.csv", index=False)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    for nm, d in [("RECESSION (out-of-fold)", rec[rec["scoring"] == "out-of-fold"]),
                  ("HEALTHCARE", hc), ("CLIMATE", cl)]:
        print(f"\n{nm}")
        cols = ["model", "horizon", "strategy", "n", "coverage",
                "wilson_lo", "wilson_hi", "mean_width", "width_ratio"]
        print(d[cols].to_string(index=False))
    print("\nSaved task7_baselines_{recession,healthcare,climate}.csv")
