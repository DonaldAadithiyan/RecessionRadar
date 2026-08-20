"""
TASK 25 — build the two new datasets with strict causal/OOF discipline.
Cached to data/domains/task25/ so downstream items are reproducible.
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_openml
OUT = "data/domains/task25"
os.makedirs(OUT, exist_ok=True)

# ── D1: insurance claim severity ───────────────────────────────────────────
p1 = f"{OUT}/insurance.csv"
if not os.path.exists(p1):
    sev = fetch_openml(data_id=41215, as_frame=True).frame
    frq = fetch_openml(data_id=41214, as_frame=True).frame
    sev["IDpol"] = sev["IDpol"].astype(np.int64)
    frq["IDpol"] = frq["IDpol"].astype(np.int64)
    # one row per claim, joined to its policy's features
    d = sev.merge(frq, on="IDpol", how="inner")
    d = d[d["ClaimAmount"] > 0].copy()
    # log severity: heavy tail, standard actuarial practice
    d["target"] = np.log1p(d["ClaimAmount"].astype(float))
    for c in ["Area", "VehBrand", "VehGas", "Region"]:
        d[c] = d[c].astype("category").cat.codes
    feats = ["Exposure", "Area", "VehPower", "VehAge", "DrivAge",
             "BonusMalus", "VehBrand", "VehGas", "Density", "Region", "ClaimNb"]
    d = d[feats + ["target"]].dropna().reset_index(drop=True)
    d.to_csv(p1, index=False)
    print(f"D1 insurance: {len(d)} claims, {len(feats)} features -> {p1}")
else:
    print(f"D1 cached: {p1}")

# ── D2: energy price spikes ────────────────────────────────────────────────
p2 = f"{OUT}/energy.csv"
if not os.path.exists(p2):
    e = fetch_openml(data_id=151, as_frame=True).frame
    for c in ["date", "day", "period", "nswprice", "nswdemand",
              "vicprice", "vicdemand", "transfer"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e = e.sort_values(["date", "period"]).reset_index(drop=True)
    # STRICTLY LAGGED predictors — no same-period information
    for lag in [1, 2, 48]:                       # 48 half-hours = 1 day
        e[f"price_lag{lag}"] = e["nswprice"].shift(lag)
        e[f"dem_lag{lag}"] = e["nswdemand"].shift(lag)
    e["price_roll48"] = e["nswprice"].shift(1).rolling(48).mean()
    e["dem_roll48"] = e["nswdemand"].shift(1).rolling(48).mean()
    e["vic_lag1"] = e["vicprice"].shift(1)
    e["transfer_lag1"] = e["transfer"].shift(1)
    e["target"] = e["nswprice"] * 100.0          # scale to a readable range
    feats = [c for c in e.columns if c.startswith(("price_lag", "dem_lag",
                                                   "price_roll", "dem_roll"))] \
        + ["vic_lag1", "transfer_lag1", "day", "period"]
    d = e[feats + ["target"]].dropna().reset_index(drop=True)
    d.to_csv(p2, index=False)
    print(f"D2 energy: {len(d)} half-hours, {len(feats)} features -> {p2}")
else:
    print(f"D2 cached: {p2}")
