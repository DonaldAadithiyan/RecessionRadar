# Task 8 — paper-ready tables

## Point-prediction comparison (mirrors Table 6's recession format)

### Healthcare (§6.3)

| Model | Origin | AUC-ROC (binary) | OOF MAE (cohort rate, pts) |
|---|---|---|---|
| LACE | van Walraven 2010 / Donzé 2013 | 0.5691 | 4.900 |
| HOSPITAL | van Walraven 2010 / Donzé 2013 | 0.5875 | 4.776 |
| ridge | internal sensitivity check | — | see §6.3 |
| gradboost | internal sensitivity check | — | see §6.3 |

Published reference points on this dataset (Emi-Johnson et al. 2026): LACE-based logistic ≈0.608, logistic ≈0.642, RF ≈0.630, XGBoost ≈0.667.

### Climate (§6.4)

| Model | Origin | OOF MAE | Test MAE | Score support (p95−p5) |
|---|---|---|---|---|
| persistence | WeatherBench baseline tier (Rasp et al. 2020) | 3.560 | 2.515 | 10.559 |
| climatology | WeatherBench baseline tier (Rasp et al. 2020) | 3.054 | 3.032 | 8.119 |

## Calibration comparison, extended to 4 underlying models per domain

(mirrors Tables 8/9; coverage % with 95% Wilson interval)

### Healthcare

| Strategy | HOSPITAL | LACE | gradboost | ridge |
|---|---|---|---|---|
| pooled_trailing | 86.99 | 86.99 | 90.41 | 88.36 |
| mondrian | 91.10 | 86.99 | 91.78 | 91.78 |
| pid_conformal | 86.99 | 87.67 | 90.41 | 88.36 |
| evt_tail | 86.99 | 86.99 | 90.41 | 89.04 |
| dtaci | 86.99 | 86.99 | 89.04 | 88.36 |
| acmcp | 81.51 | 78.08 | 89.73 | 80.82 |
| bellman_ci | 68.49 | 71.23 | 66.44 | 69.18 |
| diversity_optimal **(ours)** | 91.78 | 91.78 | 93.84 | 92.47 |

### Climate

| Strategy | climatology | gradboost | persistence | ridge |
|---|---|---|---|---|
| pooled_trailing | 90.08 | 90.95 | 90.87 | 90.95 |
| mondrian | 96.83 | 94.65 | 94.44 | 95.88 |
| pid_conformal | 89.68 | 88.89 | 90.48 | 89.71 |
| evt_tail | 90.48 | 91.36 | 90.87 | 90.53 |
| dtaci | 90.08 | 90.95 | 90.08 | 90.12 |
| acmcp | 89.68 | 88.07 | 93.25 | 88.89 |
| bellman_ci | 75.40 | 79.42 | 78.97 | 78.19 |
| diversity_optimal **(ours)** | 99.21 | 98.77 | 99.21 | 99.59 |
