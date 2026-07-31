"""
DOMAIN 3 — Climate: extreme-weather event intensity (rare high-impact months).

Dataset: NOAA Storm Events Database (public domain, U.S. federal government
work, no licence restriction on republication), yearly CSV detail files from
https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

Construction: storm events are aggregated to a monthly-by-region panel. The
target is the region-month's damaging-event intensity — the count of
significant events (those causing death, injury, or property damage) per
region-month, expressed on a comparable scale. Rare events are region-months in
the upper tail of that distribution: the climate analog of a recession month.

Predictors are strictly LAGGED aggregates of the same panel (previous months'
activity, seasonal encoding, region encoding) so nothing from the target month
leaks into its own prediction.

As with the healthcare domain, nonconformity scores are out-of-fold from the
start, and two prediction models are run (Task 2 model-agnosticism).

Outputs:
  fix-reg/domain_climate_ablation_{model}.csv
  fix-reg/domain_climate_sweep_{model}.csv
  fix-reg/domain_climate_summary.csv
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, support_width, fixed_size_ablation,
    random_draw_sweep, sweep_diagnostics, wilson_ci_from_indicator,
    block_bootstrap_ci, rolling_origin_folds, GAMMA_DEFAULT,
)

RAW_GLOB = os.environ.get("NOAA_GLOB", "data/domains/noaa/*.csv.gz")
PANEL = "data/domains/noaa_panel.csv"
OUT = "fix-reg"
SEED = 23
N_CAL = 254
N_DRAWS = 200

os.makedirs(OUT, exist_ok=True)

print("=" * 74)
print("DOMAIN 3 — CLIMATE (NOAA Storm Events, extreme-weather intensity)")
print("=" * 74)

# ── Build the monthly x region panel (cached) ─────────────────────────────────
if os.path.exists(PANEL):
    panel = pd.read_csv(PANEL)
    print(f"  Loaded cached panel: {PANEL}")
else:
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise SystemExit(f"No NOAA files matched {RAW_GLOB}")
    print(f"  Parsing {len(files)} NOAA yearly files...")

    usecols = ["BEGIN_YEARMONTH", "STATE", "EVENT_TYPE", "INJURIES_DIRECT",
               "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT",
               "DAMAGE_PROPERTY"]

    def parse_damage(x):
        """NOAA encodes damage as e.g. '5.00K', '1.2M', '3B'."""
        if not isinstance(x, str) or not x:
            return 0.0
        x = x.strip().upper()
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(x[-1], 1.0)
        try:
            return float(x[:-1]) * mult if x[-1] in "KMBT" else float(x)
        except ValueError:
            return 0.0

    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in usecols,
                        low_memory=False, compression="gzip")
        d["damage"] = d.get("DAMAGE_PROPERTY", pd.Series(dtype=str)).map(parse_damage)
        for c in ["INJURIES_DIRECT", "INJURIES_INDIRECT",
                  "DEATHS_DIRECT", "DEATHS_INDIRECT"]:
            d[c] = pd.to_numeric(d.get(c, 0), errors="coerce").fillna(0)
        d["casualties"] = (d["INJURIES_DIRECT"] + d["INJURIES_INDIRECT"]
                           + d["DEATHS_DIRECT"] + d["DEATHS_INDIRECT"])
        # A "significant" event causes casualties or material property damage.
        d["significant"] = ((d["casualties"] > 0) | (d["damage"] >= 5e4)).astype(int)
        d["ym"] = pd.to_numeric(d["BEGIN_YEARMONTH"], errors="coerce")
        d = d.dropna(subset=["ym", "STATE"])
        parts.append(d[["ym", "STATE", "significant", "casualties", "damage"]])

    ev = pd.concat(parts, ignore_index=True)
    ev["year"] = (ev["ym"] // 100).astype(int)
    ev["month"] = (ev["ym"] % 100).astype(int)

    # Regions: aggregate states into NOAA-style broad regions to keep the panel
    # dense enough that monthly rates are meaningful.
    REGION = {
        "NORTHEAST": ["MAINE", "NEW HAMPSHIRE", "VERMONT", "MASSACHUSETTS",
                      "RHODE ISLAND", "CONNECTICUT", "NEW YORK", "NEW JERSEY",
                      "PENNSYLVANIA"],
        "SOUTHEAST": ["VIRGINIA", "NORTH CAROLINA", "SOUTH CAROLINA", "GEORGIA",
                      "FLORIDA", "ALABAMA", "MISSISSIPPI", "TENNESSEE",
                      "KENTUCKY", "WEST VIRGINIA"],
        "MIDWEST": ["OHIO", "INDIANA", "ILLINOIS", "MICHIGAN", "WISCONSIN",
                    "MINNESOTA", "IOWA", "MISSOURI"],
        "PLAINS": ["NORTH DAKOTA", "SOUTH DAKOTA", "NEBRASKA", "KANSAS",
                   "OKLAHOMA", "TEXAS"],
        "WEST": ["MONTANA", "WYOMING", "COLORADO", "NEW MEXICO", "ARIZONA",
                 "UTAH", "NEVADA", "IDAHO", "WASHINGTON", "OREGON",
                 "CALIFORNIA"],
    }
    s2r = {s: r for r, ss in REGION.items() for s in ss}
    ev["region"] = ev["STATE"].astype(str).str.upper().map(s2r)
    ev = ev.dropna(subset=["region"])

    panel = (ev.groupby(["year", "month", "region"])
               .agg(n_events=("significant", "size"),
                    n_signif=("significant", "sum"),
                    casualties=("casualties", "sum"),
                    damage=("damage", "sum"))
               .reset_index())
    os.makedirs(os.path.dirname(PANEL), exist_ok=True)
    panel.to_csv(PANEL, index=False)
    print(f"  Saved panel: {PANEL}")

panel = panel.sort_values(["region", "year", "month"]).reset_index(drop=True)
print(f"  Panel rows (region-months): {len(panel):,}  "
      f"regions={panel['region'].nunique()}  "
      f"years={panel['year'].min()}-{panel['year'].max()}")

# ── Target: significant-event intensity per region-month ──────────────────────
# Expressed as significant events per 100 events-equivalent, a bounded-ish
# continuous intensity comparable in spirit to a probability series.
panel["target"] = 100.0 * panel["n_signif"] / panel["n_events"].clip(lower=1)

# ── Strictly lagged predictors (no same-month information) ────────────────────
g = panel.groupby("region")
for lag in [1, 2, 3, 6, 12]:
    panel[f"tgt_lag{lag}"] = g["target"].shift(lag)
    panel[f"cnt_lag{lag}"] = g["n_events"].shift(lag)
panel["tgt_roll3"] = g["target"].shift(1).rolling(3).mean().reset_index(0, drop=True)
panel["tgt_roll12"] = g["target"].shift(1).rolling(12).mean().reset_index(0, drop=True)
panel["cas_lag1"] = g["casualties"].shift(1)
panel["dmg_lag1"] = np.log1p(g["damage"].shift(1))
panel["sin_m"] = np.sin(2 * np.pi * panel["month"] / 12)
panel["cos_m"] = np.cos(2 * np.pi * panel["month"] / 12)
reg_d = pd.get_dummies(panel["region"], prefix="reg", dtype=float)
panel = pd.concat([panel, reg_d], axis=1)

feat_cols = ([c for c in panel.columns if c.startswith(("tgt_lag", "cnt_lag", "tgt_roll"))]
             + ["cas_lag1", "dmg_lag1", "sin_m", "cos_m"]
             + list(reg_d.columns))

panel = panel.dropna(subset=feat_cols + ["target"]).reset_index(drop=True)
print(f"  Usable region-months after lagging: {len(panel):,}")

y_all = panel["target"].values.astype(float)
X_all = panel[feat_cols].values.astype(float)

# Rare event = upper-decile intensity region-month (a damaging-weather month).
RARE_THRESH = float(np.percentile(y_all, 90))
is_rare_all = y_all >= RARE_THRESH
print(f"  Intensity: mean={y_all.mean():.2f}  p90={RARE_THRESH:.2f}  max={y_all.max():.2f}")
print(f"  Rare region-months: {is_rare_all.sum():,} ({100*is_rare_all.mean():.1f}%)")

# ── Temporal split: last 15% of the panel is the test stream ─────────────────
panel["t"] = panel["year"] * 12 + panel["month"]
order = np.argsort(panel["t"].values, kind="stable")
X_all, y_all, is_rare_all = X_all[order], y_all[order], is_rare_all[order]

n = len(y_all)
n_test = int(0.15 * n)
pool_idx = np.arange(0, n - n_test)
test_idx = np.arange(n - n_test, n)
X_pool, y_pool = X_all[pool_idx], y_all[pool_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
rare_pool = is_rare_all[pool_idx]
N_CAL_EFF = min(N_CAL, int(0.6 * len(pool_idx)))
print(f"  Calibration pool: {len(y_pool):,} ({rare_pool.sum()} rare) | "
      f"Test stream: {len(y_test):,} | N_cal={N_CAL_EFF}")

MODELS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "gradboost": lambda: HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.06, max_depth=4, random_state=SEED),
}

scaler = StandardScaler().fit(X_pool)
Xp_s, Xt_s = scaler.transform(X_pool), scaler.transform(X_test)

summary_rows = []
for name, factory in MODELS.items():
    print("\n" + "-" * 74)
    print(f"MODEL: {name}")
    print("-" * 74)
    Xp = Xp_s if name == "ridge" else X_pool
    Xt = Xt_s if name == "ridge" else X_test

    # ── Out-of-fold scores via rolling-origin folds (temporal safety) ─────────
    oof = np.full(len(y_pool), np.nan)
    for tr, te in rolling_origin_folds(len(y_pool), n_folds=5):
        m = factory()
        m.fit(Xp[tr], y_pool[tr])
        oof[te] = m.predict(Xp[te])

    m_full = factory()
    m_full.fit(Xp, y_pool)
    pred_test = m_full.predict(Xt)

    scores_all = np.abs(oof - y_pool)
    print(f"  OOF MAE: {np.nanmean(scores_all):.2f}  "
          f"support(p95-p5)={support_width(scores_all):.2f}  "
          f"(scored {np.isfinite(scores_all).sum()}/{len(scores_all)})")

    rare_counts = [0, 1, 4, 8, 16, 32, 64]
    abl = fixed_size_ablation(scores_all, rare_pool, y_test, pred_test,
                              rare_counts, N=N_CAL_EFF, seed=SEED)
    abl.insert(0, "domain", "Climate")
    abl.insert(1, "model", name)
    abl.to_csv(f"{OUT}/domain_climate_ablation_{name}.csv", index=False)
    print(f"\n  Fixed-size ablation (N={N_CAL_EFF}, varying rare-event count only):")
    print(abl[["Rare_Units", "Composition_pct", "Coverage",
               "Wilson_lo", "Wilson_hi", "Support_width"]].to_string(index=False))

    sweep = random_draw_sweep(scores_all, rare_pool, y_test, pred_test,
                             N=N_CAL_EFF, n_draws=N_DRAWS, seed=SEED)
    sweep.insert(0, "model", name)
    sweep.to_csv(f"{OUT}/domain_climate_sweep_{name}.csv", index=False)

    row = sweep_diagnostics(sweep, domain="Climate",
                            extra=dict(model=name, pool=len(y_pool),
                                       N=N_CAL_EFF, test=len(y_test),
                                       base_rate_pct=round(float(np.mean(y_all)), 2)))
    summary_rows.append(row)
    print(f"\n  rho(support,cov)={row['rho_supp']}  R2={row['R2_supp']}  |  "
          f"rho(rare,cov)={row['rho_rare']}  R2={row['R2_rare']}")
    print(f"  within-tertile rho(rare,cov|support fixed) = "
          f"{row['within_tertile_rho_rare']}")

    tail = np.arange(len(y_pool) - N_CAL_EFF, len(y_pool))
    s_tail = scores_all[tail]
    s_tail = s_tail[~np.isnan(s_tail)]
    cov_arr, _, w_arr = run_aci(y_test, pred_test, s_tail, gamma=GAMMA_DEFAULT)
    cov, w = coverage_and_width(cov_arr, w_arr)
    wlo, whi = wilson_ci_from_indicator(cov_arr)
    blo, bhi = block_bootstrap_ci(cov_arr, block=12)  # monthly autocorrelation
    print(f"  trailing-N coverage={cov:.2f}%  Wilson[{wlo:.2f},{whi:.2f}]  "
          f"block-bootstrap[{blo:.2f},{bhi:.2f}]  width={w:.2f}")
    summary_rows[-1].update(dict(
        trailing_cov=round(cov, 2), trailing_wilson_lo=round(wlo, 2),
        trailing_wilson_hi=round(whi, 2), trailing_boot_lo=round(blo, 2),
        trailing_boot_hi=round(bhi, 2), trailing_width=round(w, 2)))

pd.DataFrame(summary_rows).to_csv(f"{OUT}/domain_climate_summary.csv", index=False)
print(f"\nSaved {OUT}/domain_climate_summary.csv")
