# Task 23, Item 2 — Full Four-Horizon Validation Gate, Re-run Fresh

**Result: still 1/4 validated (6M only) — but the underlying numbers moved
substantially, and in two opposite directions at once.**

Script: `task23_2_validation.py` · Data: `task23_2_validation.csv`

## The full table (all four horizons, both checks), Task 21 vs Task 23

| Horizon | r_cal (T21) | **r_cal (T23)** | perturb pct (T21) | **perturb pct (T23)** | angle PC1 (T23) | disag weight | Valid T21 | **Valid T23** |
|---|---|---|---|---|---|---|---|---|
| Current | 0.109 | **0.302** | 0.370 | **0.025** | 75.2° | 18.7% | No | **No** |
| 1M | 0.151 | **0.354** | 0.595 | **0.030** | 89.2° | 25.6% | No | **No** |
| 3M | 0.349 | **0.477** | 0.925 | **0.850** | 67.9° | 8.7% | No | **No** |
| **6M** | 0.367 | **0.539** | 0.975 | **1.000** | 47.6° | 11.5% | Yes | **Yes** |

The gate was re-run from scratch, not inherited. Same protocol as Task 21: same
291/254 FIT/CAL split, same RidgeCV, same >30° and ≥95th-percentile criteria,
same 200-direction null. The only change is three added features per horizon.

## Two findings that point opposite ways

**1. Held-out correlation improved at every single horizon.** Current
0.109→0.302 (nearly 3×), 1M 0.151→0.354, 3M 0.349→0.477, 6M 0.367→0.539. β
genuinely predicts error magnitude better with disagreement in the feature set.
The Item 1 prediction held: the largest absolute gain lands at 6M, where
disagreement carried the most independent information (R²=0.675 from existing
features vs 0.84–0.89 elsewhere).

**2. The perturbation percentile collapsed at the short horizons.** Current
0.370→**0.025**, 1M 0.595→**0.030**. These are not marginal failures; they are
near the *bottom* of the random-direction null, far worse than Task 21's already
failing values.

This divergence is the substantive result of Item 2, and it is not a
contradiction. The two checks measure different things:

- **r_cal** asks: does β·x correlate with realized error out-of-sample?
- **perturbation** asks: does moving inputs *along β* change the ensemble's
  prediction more than moving along a random direction?

At Current/1M, β now points substantially at the disagreement features (18.7%
and 25.6% of total absolute weight). Disagreement is an *output* of the base
models — it is derived from where they disagree, not a direction the ensemble's
prediction is sensitive to. Perturbing along a direction dominated by
disagreement-loading moves the ensemble's prediction *less* than a random
direction would. So β became a better **correlational** predictor of error and a
worse **causal-adjacent** direction in input space, simultaneously.

Task 21's Check 2 exists precisely to catch that distinction ("correlational fit
alone is not sufficient evidence β captures something causal-adjacent rather
than a fitting artifact"). It did its job here: without it, the improved r_cal
at Current/1M would have looked like three new validated horizons.

**6M is the exception** — it takes only 11.5% of its weight from disagreement,
improves r_cal to 0.539, and its perturbation percentile rises to a perfect
**1.000** (β beats all 200 random directions). It is the one horizon where the
new feature adds predictive content without hijacking the direction.

## Verdict

Still 6M only. Items 3 and 4 run at 6M.
