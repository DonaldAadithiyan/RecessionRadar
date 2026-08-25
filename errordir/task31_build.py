"""
TASK 31 — build candidate domains for concept-drift screening.

C1 FRAUD (credit-card): 284,807 transactions. NOTE: the OpenML copy has NO Time
column, so chronological ordering is unavailable. Row order is used as a proxy
for arrival order (the dataset is distributed in time order), and this
limitation is recorded explicitly in the screening report -- if the proxy is
wrong the screen is invalid, so this counts against the candidate.
Target: log1p(Amount) among FRAUD transactions is too small (492 rows), so the
target is log1p(Amount) over all transactions, with fraud flag as a feature.

C2 EPIDEMIC (OWID COVID): daily national panel. Concept drift is textbook here --
the relationship between cases/mobility/testing and DEATHS changed across waves
(variants, vaccination, accumulated immunity, policy). Target: deaths per
million; predictors are STRICTLY LAGGED case/hospital/policy series.
"""
import os, io, urllib.request, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "data/domains/task31"
os.makedirs(OUT, exist_ok=True)

# ── C1 fraud ───────────────────────────────────────────────────────────────
p1 = f"{OUT}/fraud.csv"
if not os.path.exists(p1):
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=1597, as_frame=True).frame
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"]).reset_index(drop=True)
    df["target"] = np.log1p(df["Amount"])
    feats = [f"V{i}" for i in range(1, 29)] + ["Class"]
    d = df[feats + ["target"]].dropna().reset_index(drop=True)
    d.to_csv(p1, index=False)
    print(f"C1 fraud: {len(d)} rows, {len(feats)} features -> {p1}")
else:
    print(f"C1 cached: {p1}")

# ── C2 epidemic ────────────────────────────────────────────────────────────
p2 = f"{OUT}/epidemic.csv"
if not os.path.exists(p2):
    u = ("https://raw.githubusercontent.com/owid/covid-19-data/master/"
         "public/data/owid-covid-data.csv")
    raw = urllib.request.urlopen(u, timeout=300).read()
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    cols = ["date", "location", "new_cases_smoothed_per_million",
            "new_deaths_smoothed_per_million", "reproduction_rate",
            "icu_patients_per_million", "hosp_patients_per_million",
            "positive_rate", "stringency_index",
            "people_vaccinated_per_hundred"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"])
    # countries with dense reporting only
    dense = (df.groupby("location")["new_deaths_smoothed_per_million"]
               .apply(lambda s: s.notna().sum()))
    keep = dense[dense > 700].index.tolist()
    df = df[df["location"].isin(keep)].sort_values(["location", "date"])
    g = df.groupby("location")
    # STRICTLY LAGGED predictors — no same-day information
    for lag in [7, 14, 21]:
        df[f"cases_lag{lag}"] = g["new_cases_smoothed_per_million"].shift(lag)
        df[f"pos_lag{lag}"] = g["positive_rate"].shift(lag)
    for c, nm in [("reproduction_rate", "rt"), ("icu_patients_per_million", "icu"),
                  ("hosp_patients_per_million", "hosp"),
                  ("stringency_index", "strin"),
                  ("people_vaccinated_per_hundred", "vax")]:
        if c in df.columns:
            df[f"{nm}_lag7"] = g[c].shift(7)
    df["deaths_lag14"] = g["new_deaths_smoothed_per_million"].shift(14)
    df["deaths_roll28"] = g["new_deaths_smoothed_per_million"].shift(7).rolling(28).mean().reset_index(0, drop=True)
    df["target"] = df["new_deaths_smoothed_per_million"]
    feats = [c for c in df.columns if c.endswith(("_lag7", "_lag14", "_lag21",
                                                  "_roll28"))]
    d = df[["date"] + feats + ["target"]].dropna()
    d = d.sort_values("date").reset_index(drop=True)
    d.to_csv(p2, index=False)
    print(f"C2 epidemic: {len(d)} rows, {len(feats)} features -> {p2}")
else:
    print(f"C2 cached: {p2}")
