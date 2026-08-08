"""
TASK 17, Item 2 — stratified and mixed calibration baselines.

THE QUESTION, PRECISELY. Every calibration strategy tested so far is either a
full conformal method (Mondrian, DtACI, PID, AcMCP, Bellman CI, CPTC) or one of
the extremes (trailing, rare-heavy, tail-extreme). There is a real gap in
between: does ANY form of spreading the calibration set across the score range
capture most of the benefit, or is it specifically the EXTREMES that matter?

This is the sharpest available test of whether the paper's selector is uniquely
motivated by tail-reach or is just one instance of a broader, less interesting
"spread your calibration set" principle. If decile-stratified sampling closes
most of the gap, that reframes the selector and the paper should say so.

Strategies added (spec):
  decile_stratified   partition the pool into 10 score deciles, take N/10 each
  quintile_stratified same at 5 bins (coarser variant)
  tail_plus_recency   50% most recent months + 50% from the two score tails
                      (using the existing alternating-tail rule for that half)

Reported alongside the existing reference points (trailing, diversity-optimal)
in the same format as the paper's calibration tables.

A DIAGNOSTIC THE SPEC IMPLIES BUT DOES NOT NAME: since Task 10 showed coverage
is driven by the calibration set's (1-alpha) quantile, each strategy's Q_C is
also reported. If stratified sampling closes the coverage gap, its Q_C should
also approach the selector's — and if it closes the coverage gap WITHOUT
matching Q_C, that would contradict the paper's mechanism and needs reporting.

Outputs:
  fix-reg/task17_2_stratified_baselines.csv
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

from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, support_width, wilson_ci_from_indicator,
    block_bootstrap_ci, GAMMA_DEFAULT, ALPHA_TARGET,
)
import selector_lib as SEL  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
LABELS = ["Current", "1M", "3M", "6M"]


# ── selection rules ─────────────────────────────────────────────────────────

def sel_trailing(scores, N, **kw):
    v = np.where(np.isfinite(scores))[0]
    return v[-N:] if len(v) >= N else v


def sel_stratified(scores, N, n_bins=10, seed=0, **kw):
    """
    Partition the pool into n_bins equal-count score bins and take N/n_bins from
    each. This spreads the calibration set across the whole score range without
    preferentially loading the extremes.
    """
    rng = np.random.default_rng(seed)
    v = np.where(np.isfinite(scores))[0]
    s = scores[v]
    order = v[np.argsort(s)]
    bins = np.array_split(order, n_bins)
    per = N // n_bins
    chosen = []
    for b in bins:
        k = min(per, len(b))
        chosen.append(rng.choice(b, size=k, replace=False))
    chosen = np.concatenate(chosen)
    # top up to exactly N from whatever remains, uniformly
    if len(chosen) < N:
        rest = np.setdiff1d(order, chosen)
        extra = rng.choice(rest, size=min(N - len(chosen), len(rest)),
                           replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen[:N])


def sel_tail_plus_recency(scores, N, seed=0, **kw):
    """50% most-recent months + 50% from the two score tails."""
    v = np.where(np.isfinite(scores))[0]
    half = N // 2
    recent = v[-half:]
    remaining = np.setdiff1d(v, recent)
    if len(remaining) == 0:
        return np.sort(recent)
    # alternating-tail rule over what's left (the selector's own half)
    sub = scores[remaining]
    order = remaining[np.argsort(sub)]
    chosen, lo, hi, take_low = [], 0, len(order) - 1, True
    while len(chosen) < min(N - half, len(order)) and lo <= hi:
        if take_low:
            chosen.append(order[lo]); lo += 1
        else:
            chosen.append(order[hi]); hi -= 1
        take_low = not take_low
    return np.sort(np.concatenate([recent, np.array(chosen, dtype=int)]))


def sel_diversity_optimal(scores, N, **kw):
    return SEL.support_width_selector(scores, N)


STRATEGIES = {
    "pooled_trailing": sel_trailing,
    "decile_stratified": lambda s, N, **k: sel_stratified(s, N, 10, **k),
    "quintile_stratified": lambda s, N, **k: sel_stratified(s, N, 5, **k),
    "tail_plus_recency": sel_tail_plus_recency,
    "diversity_optimal": sel_diversity_optimal,
}


def evaluate(domain, model, horizon, scores, y_te, p_te, block, base=None):
    rows = []
    for name, fn in STRATEGIES.items():
        idx = fn(scores, N_FIX, seed=11)
        cal = np.asarray(scores)[idx]
        cal = cal[np.isfinite(cal)]
        if len(cal) < 2:
            continue
        covered, _, widths = run_aci(y_te, p_te, cal, gamma=GAMMA_DEFAULT)
        cov, w = coverage_and_width(covered, widths)
        lo, hi = wilson_ci_from_indicator(covered)
        blo, bhi = block_bootstrap_ci(covered, block=block)
        rows.append(dict(domain=domain, model=model, horizon=horizon,
                         strategy=name, n_cal=int(len(cal)),
                         n_test=int(np.isfinite(
                             np.asarray(covered, float)).sum()),
                         coverage=round(cov, 2),
                         wilson_lo=round(lo, 2), wilson_hi=round(hi, 2),
                         boot_lo=round(blo, 2), boot_hi=round(bhi, 2),
                         mean_width=round(w, 3),
                         support=round(support_width(cal), 3),
                         Q_C=round(float(np.quantile(cal, 1 - ALPHA_TARGET)), 3)))
    return rows


print("=" * 100)
print("TASK 17 Item 2 — stratified / mixed calibration baselines")
print("=" * 100)
print("  Question: does generic 'spread' capture the benefit, or is it")
print("  specifically the EXTREMES that matter?")

all_rows = []

import task_oof_and_probit as T  # noqa: E402
print("\n" + "-" * 100)
print("RECESSION (primary testbed, out-of-fold)")
print("-" * 100)
for h_idx, h in enumerate(LABELS):
    s = np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx])
    rows = evaluate("Recession", "stacking-chain", h, s,
                    T.y_test[:, h_idx], T.preds_test[:, h_idx], block=12)
    all_rows += rows
    print(f"\n  {h}:")
    base = next((r for r in rows if r["strategy"] == "pooled_trailing"), None)
    div = next((r for r in rows if r["strategy"] == "diversity_optimal"), None)
    for r in rows:
        gap_note = ""
        if base and div and div["coverage"] > base["coverage"]:
            closed = ((r["coverage"] - base["coverage"]) /
                      (div["coverage"] - base["coverage"]) * 100)
            gap_note = f"  closes {closed:5.1f}% of the selector's gain"
        print(f"    {r['strategy']:22s} cov={r['coverage']:6.2f}% "
              f"[{r['wilson_lo']:5.2f},{r['wilson_hi']:5.2f}] "
              f"w={r['mean_width']:8.2f}  Q_C={r['Q_C']:7.2f}{gap_note}")

import runpy  # noqa: E402
print("\n" + "-" * 100)
print("HEALTHCARE / CLIMATE")
print("-" * 100)
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    all_rows += evaluate("Healthcare", name, "30-day", g["SCORES"][name],
                         g["Y_TEST"], g["PREDS_TEST"][name], block=1)
gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    all_rows += evaluate("Climate", name, "region-month", gc["SCORES"][name],
                         gc["Y_TEST"], gc["PREDS_TEST"][name], block=12)

R = pd.DataFrame(all_rows)
R.to_csv(f"{OUT}/task17_2_stratified_baselines.csv", index=False)

# ── How much of the selector's gain does generic spreading capture? ─────────
print("\n" + "=" * 100)
print("HOW MUCH OF THE SELECTOR'S GAIN DOES EACH STRATEGY CAPTURE?")
print("=" * 100)
frac_rows = []
for (d, m, h), grp in R.groupby(["domain", "model", "horizon"]):
    base = grp[grp.strategy == "pooled_trailing"]
    div = grp[grp.strategy == "diversity_optimal"]
    if base.empty or div.empty:
        continue
    b, v = base.iloc[0], div.iloc[0]
    denom = v["coverage"] - b["coverage"]
    if abs(denom) < 1e-9:
        continue
    for name in ["decile_stratified", "quintile_stratified",
                 "tail_plus_recency"]:
        r = grp[grp.strategy == name]
        if r.empty:
            continue
        r = r.iloc[0]
        frac_rows.append(dict(
            domain=d, model=m, horizon=h, strategy=name,
            base_cov=b["coverage"], strat_cov=r["coverage"],
            divopt_cov=v["coverage"],
            frac_of_gain=round((r["coverage"] - b["coverage"]) / denom, 3),
            width_vs_divopt=round(r["mean_width"] / v["mean_width"], 3)
            if v["mean_width"] else np.nan,
            QC_vs_divopt=round(r["Q_C"] / v["Q_C"], 3) if v["Q_C"] else np.nan))
F = pd.DataFrame(frac_rows)
F.to_csv(f"{OUT}/task17_2_gain_fractions.csv", index=False)
print(F.to_string(index=False))

for name in ["decile_stratified", "quintile_stratified", "tail_plus_recency"]:
    sub = F[F.strategy == name]
    if sub.empty:
        continue
    print(f"\n  {name}: median {100*sub.frac_of_gain.median():.1f}% of the "
          f"selector's gain, median Q_C ratio {sub.QC_vs_divopt.median():.3f}")

print(f"\nSaved {OUT}/task17_2_stratified_baselines.csv, "
      f"task17_2_gain_fractions.csv")
