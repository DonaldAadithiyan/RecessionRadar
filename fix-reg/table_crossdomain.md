# Cross-domain generalization of the calibration-diversity finding

Each row: 200 random fixed-size calibration draws; Spearman rho of ACI
coverage against calibration-score support width (p95-p5) vs against
rare-event count; and the redundancy check (rho of rare-count with
coverage *within* support-width tertiles).

| Domain | Model | Scoring | rho(diversity) | rho(rare-count) | rho(rare \| diversity fixed) | mean cov % |
|---|---|---|---|---|---|---|
| Macro-recession (US, 3M) | stacking-chain | in-sample | 0.683 | 0.331 | 0.312 | 87.28 |
| Macro-recession (US, 6M) | stacking-chain | in-sample | 0.913 | 0.098 | 0.045 | 72.69 |
| Healthcare (readmission) | ridge | out-of-fold | 0.687 | 0.491 | 0.184 | 89.09 |
| Healthcare (readmission) | gradboost | out-of-fold | 0.632 | 0.340 | 0.063 | 90.13 |
| Climate (storm intensity) | ridge | out-of-fold | 0.675 | 0.565 | 0.359 | 92.57 |
| Climate (storm intensity) | gradboost | out-of-fold | 0.638 | 0.431 | 0.256 | 92.16 |

**Reading:** in every domain and under every underlying model, support-width
diversity is the stronger predictor of ACI coverage, and rare-event count
attenuates once diversity is held fixed. The recession row is the paper's
original testbed; healthcare and climate are the new domains.
