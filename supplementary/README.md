# Supplementary Material — RecessionRadar (IJCAI GlobalSouthAI)

Reproduces Table 1 (ensemble MAE), the ablation baselines (Table 2 numbers),
the ACI experiments (Table 3), and the calibration-composition sweep (P0.2).

## Contents

```
supplementary/
├── data/
│   ├── raw/                     13 raw FRED series CSVs (see table below)
│   └── processed/
│       └── feature_selected_reg_full.csv   700 monthly obs × 40 features + 4 targets + date
├── models/
│   └── full_chain_stacking.pkl  saved Stage-2 ensemble (CatBoost+LGBM+RF, ElasticNet meta, RegressorChain)
├── scripts/
│   ├── scenario_test.py         counterfactual rate-shock simulation (paper scenario table)
│   ├── ablation_baselines.py    XGB-indep / MOR-XGB / MOR-XGB-joint baselines + DM tests
│   ├── aci_experiments.py       ACI gamma grid + calibration-window experiments (Table 3)
│   └── aci_composition_sweep.py 5-point calibration-composition sweep (P0.2)
└── expected_outputs/            reference CSVs to diff your run against
```

## FRED series (identifiers, meaning, span)

All series pulled from FRED (https://fred.stlouisfed.org/series/<ID>), monthly,
1967-02 → 2025-06 (target series ends 2025-05).

| FRED ID | Meaning | Role |
|---|---|---|
| `RECPROUSM156N` | Smoothed U.S. Recession Probabilities | **Target** (Current; 1M/3M/6M = shift(−1/−3/−6)) |
| `DTB3` | 3-Month Treasury Bill rate | feature |
| `DTB6` | 6-Month Treasury Bill rate | feature |
| `DTB1YR` | 1-Year Treasury Bill rate | feature |
| `IRLTLT01USM156N` | 10-Year Long-Term Govt Bond Yield | feature (yield-curve spread) |
| `CPIAUCSL` | CPI, All Urban Consumers | feature |
| `INDPRO` | Industrial Production Index | feature |
| `PCU3312103312100` | PPI (industry) | feature |
| `A939RX0Q048SBEA` | Real GDP per capita (quarterly→monthly interp.) | feature |
| `SPASTT01USM661N` | Share Price Index | feature |
| `UMCSENT` | U. Michigan Consumer Sentiment (CSI) | feature |
| `UNRATE` | Unemployment Rate | feature |
| `USALOLITOAASTSAM` | OECD Composite Leading Indicator (US) | feature |

**Train/test split:** `date < 2020-01-01` → 635 train rows; `date ≥ 2020-01-01`
→ 65 test rows.

## Environment

The saved model was pickled with **scikit-learn 1.5.2** — pin it or unpickling
may fail. The project's full `requirements.txt` is a frozen dev env (TensorFlow,
Torch, Ray, Prophet); **none of that is needed** for the reproductions here.
Minimal set (verified in a clean venv):

```bash
python3 -m venv venv && source venv/bin/activate
pip install "scikit-learn==1.5.2" numpy pandas scipy catboost lightgbm xgboost matplotlib
```

## How to run

Run from inside `scripts/` (paths are relative to that dir):

```bash
cd scripts
python aci_composition_sweep.py      # ~30s  → expected_outputs/aci_composition_sweep_repro.csv
python ablation_baselines.py         # SLOW: re-runs Optuna (see note) ~10 min
python scenario_test.py              # ~20s, writes a PDF report
python aci_experiments.py            # ~1 min
```

## Expected runtime & what reproduces

| Step | Runtime | Reproduces exactly? |
|---|---|---|
| Table 1 ensemble MAE (from saved pkl) | ~15 s | **Yes, to 4 dp** (6.83/5.63/7.73/10.17) |
| ACI composition sweep (P0.2) | ~30 s | **Yes** (re-slices saved preds; no training) |
| ACI experiments (Table 3) | ~1 min | **Yes** (uses saved pkl) |
| Ablation baselines (Table 2) | ~10 min | **Yes**, but re-runs 40-trial Optuna ×5 searches; deterministic (seed=42) |

## What does NOT reproduce from raw data alone

- **`feature_selected_reg_full.csv` is shipped pre-built.** The raw→processed
  feature-engineering + RF feature-selection pipeline lives in project
  notebooks not included here; the processed CSV is provided so the modeling
  steps reproduce. Regenerating it from `data/raw/` requires those notebooks.
- **The ensemble model is shipped as a pickle**, not retrained here. Retraining
  (`ensemble.ipynb` in the main repo) needs catboost/lightgbm and is
  deterministic given seeds, but is not part of this package.
- Scripts were path-patched for this folder layout; the versions in the main
  repo use repo-relative paths.
