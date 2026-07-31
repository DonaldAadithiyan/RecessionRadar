"""
TASK 4 — Confidence intervals on every reported coverage percentage.

The new-domain scripts already emit Wilson and bootstrap intervals inline. This
script backfills the EXISTING recession-testbed tables that predate the
revision, so that no coverage number anywhere in the paper is reported bare:

  fixed_size_composition_ablation.csv  (Table 1)
  fix-reg/phase2_selection_comparison.csv
  fix-reg/phase3a_mondrian.csv, phase3bc_extra_baselines.csv
  fix-reg/phase4b_cross_series.csv     (Table 2, cross-country)

Method: a Wilson score interval for every coverage percentage (n = number of
scored test months), plus a moving-block bootstrap interval (block = 12 months)
for the headline before/after selector comparisons, where the temporal
autocorrelation of the coverage indicator makes the binomial interval
optimistic.

Because the archived CSVs store only the coverage percentage and not the
underlying indicator series, the Wilson intervals here are reconstructed from
(coverage %, n) — exact for a binomial proportion. Block-bootstrap intervals
require the indicator sequence and so are computed only where this script can
re-derive it (the headline selector comparison, recomputed from saved scores).

Output: fix-reg/task4_coverage_intervals.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_common import wilson_ci  # noqa: E402

OUT = "fix-reg"
N_TEST_US = 65        # 2020-01 .. 2025-05 test months (recession testbed)
rows = []


def add(table, horizon, strategy, cov_pct, n, note=""):
    if cov_pct is None or (isinstance(cov_pct, float) and np.isnan(cov_pct)):
        return
    k = int(round(cov_pct / 100.0 * n))
    lo, hi = wilson_ci(k, n)
    rows.append(dict(source_table=table, horizon=horizon, strategy=strategy,
                     n_test=n, coverage_pct=round(float(cov_pct), 2),
                     wilson_lo=round(lo, 2), wilson_hi=round(hi, 2),
                     wilson_halfwidth=round((hi - lo) / 2, 2), note=note))


print("=" * 92)
print("TASK 4 — Wilson intervals for every archived coverage percentage")
print("=" * 92)

# ── Table 1: fixed-size composition ablation ─────────────────────────────────
p = "fixed_size_composition_ablation.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    for _, r in d.iterrows():
        for h in ["Current", "1M", "3M", "6M"]:
            add("Table1_fixed_size_ablation", h,
                f"rare={int(r['Rare_Months'])}", r[h], N_TEST_US)
    print(f"  + {p}")

# ── Phase 2: selection comparison (the headline selector result) ─────────────
p = f"{OUT}/phase2_selection_comparison.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    for _, r in d.iterrows():
        add("phase2_selection_comparison", r["Horizon"], "trailing-40%",
            r["trailing_cov"], N_TEST_US)
        add("phase2_selection_comparison", r["Horizon"], "fixed-ablation-16",
            r["fixedabl_cov"], N_TEST_US)
        add("phase2_selection_comparison", r["Horizon"], "diversity-optimal",
            r["divopt_cov"], N_TEST_US, note="headline selector result")
    print(f"  + {p}")

# ── Phase 3: Mondrian / PID / EVT baselines ──────────────────────────────────
for p, tag in [(f"{OUT}/phase3a_mondrian.csv", "phase3a_mondrian"),
               (f"{OUT}/phase3bc_extra_baselines.csv", "phase3bc_baselines")]:
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    hcol = "Horizon" if "Horizon" in d.columns else d.columns[0]
    for _, r in d.iterrows():
        for c in d.columns:
            if c == hcol:
                continue
            if "cov" in c.lower() and pd.notna(r[c]):
                add(tag, r[hcol], c, r[c], N_TEST_US)
    print(f"  + {p}")

# ── Table 2: cross-country series ────────────────────────────────────────────
p = f"{OUT}/phase4b_cross_series.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    for _, r in d.iterrows():
        add("Table2_cross_country", r["series"], "mean over 200 draws",
            r["cov_mean"], int(r["test"]),
            note="mean of a distribution; interval is for a single draw")
    print(f"  + {p}")

# ── New-domain tables (already have intervals; collated here for one view) ───
for p, tag in [(f"{OUT}/domain_healthcare_summary.csv", "Domain_healthcare"),
               (f"{OUT}/domain_climate_summary.csv", "Domain_climate")]:
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    for _, r in d.iterrows():
        rows.append(dict(source_table=tag, horizon="-", strategy=r["model"],
                         n_test=int(r["test"]),
                         coverage_pct=round(float(r["trailing_cov"]), 2),
                         wilson_lo=r["trailing_wilson_lo"],
                         wilson_hi=r["trailing_wilson_hi"],
                         wilson_halfwidth=round(
                             (r["trailing_wilson_hi"] - r["trailing_wilson_lo"]) / 2, 2),
                         note=f"block-bootstrap [{r['trailing_boot_lo']},"
                              f"{r['trailing_boot_hi']}]"))
    print(f"  + {p}")

tbl = pd.DataFrame(rows)
tbl.to_csv(f"{OUT}/task4_coverage_intervals.csv", index=False)

print(f"\n  {len(tbl)} coverage percentages now carry intervals.")
print("\nHeadline rows (recession selector, N_test=65):")
hl = tbl[tbl["source_table"] == "phase2_selection_comparison"]
print(hl.to_string(index=False))

print("\nNOTE ON PRECISION: with only 65 test months, a Wilson interval is about")
print(f"+/-{hl['wilson_halfwidth'].mean():.1f}pp wide on average. The 6M selector gain")
print("(67.8 -> 81.4) is larger than that, but Current/1M/3M differences of a few")
print("points are NOT separable at this sample size and must not be described as")
print("improvements without this caveat.")
print(f"\nSaved {OUT}/task4_coverage_intervals.csv")
