# Task 9B — Bellman Conformal Inference, Tuned

**Verdict: tuning helps BCI a great deal — up to +22pp — but it still does not
beat the diversity-optimal selector anywhere (0 of 8 comparisons). The paper's
"no baseline tested beats the selector" framing survives, and is now stated
against a *fairly tuned* BCI rather than an untuned one.**

Script: `fix-reg/task9b_bci_tuning.py`. Data: `task9b_bci_tuning.csv`.

---

## Tuning protocol (test data never touched)

- **Validation slice comes from the calibration pool only.** For each
  (domain, model, horizon) the scored pool is split temporally: the first 80%
  supplies calibration scores, the final 20% becomes a held-out validation
  stream. The real test window plays no part in parameter selection.
- **Grid:** `lambda_init ∈ {1, 5, 20, 50}`, `lambda_max ∈ {50, 200, 500, 2000}`,
  `gamma ∈ {0.4, 0.8, 1.6, 3.2}` — the last spanning 0.5×–4× the authors'
  default of 0.8, as the task specified. 60 valid configurations per cell
  (combinations with `lambda_max ≤ lambda_init` are skipped).
- **Objective:** minimise |validation coverage − 90%|, ties broken toward the
  *narrower* mean interval, so the search cannot buy coverage with vacuous
  width.
- Selected parameters are then applied unchanged to the same out-of-fold test
  evaluation every other strategy receives.

## Tuned vs untuned

| Domain | Model | Horizon | Untuned cov | Tuned cov | Δ | Untuned w | Tuned w |
|---|---|---|---|---|---|---|---|
| Recession | stacking-chain | Current | 80.00 | **89.23** | +9.23 | 4.56 | 20.95 |
| Recession | stacking-chain | 1M | 79.69 | 82.81 | +3.12 | 3.96 | 8.22 |
| Recession | stacking-chain | 3M | 61.29 | **83.87** | +22.58 | 4.88 | 13.21 |
| Recession | stacking-chain | 6M | 49.15 | 57.63 | +8.48 | 4.55 | 27.04 |
| Healthcare | ridge | 30-day | 69.18 | **86.99** | +17.81 | 12.36 | 20.93 |
| Healthcare | gradboost | 30-day | 66.44 | **88.36** | +21.92 | 11.81 | 21.85 |
| Climate | ridge | region-month | 78.19 | **91.36** | +13.17 | 7.78 | 10.75 |
| Climate | gradboost | region-month | 79.42 | **90.95** | +11.53 | 7.88 | 11.41 |

**Tuning matters enormously — this vindicates the caveat the paper already
carried.** Gains range from +3.1pp to +22.6pp, and in climate and healthcare
tuned BCI moves from badly undercovering to essentially nominal (90.95–91.36% in
climate; 86.99–88.36% in healthcare). Reporting untuned BCI alone would have
substantially understated the method.

The tuned parameters are also consistent in a way that suggests the search found
real structure rather than noise: 5 of 8 cells select
`lambda_init=50, lambda_max=200, gamma=3.2` — much higher initial λ and much
faster adaptation than the authors' defaults (5 / 500 / 0.8).

## The exception: recession six months stays broken

Recession 6M is the one cell where tuning does not rescue BCI: 49.15 → 57.63%,
still ~32pp below nominal, and it gets there by inflating width 6× (4.55 →
27.04). This is the paper's hardest horizon, and it is the one place BCI's
length-penalised control cannot find a setting that both covers and stays tight.

Consistent with the rest of the paper's findings: the six-month deficit is a
calibration-*diversity* problem, and an algorithm that optimises interval length
subject to a coverage constraint cannot manufacture tail coverage its
calibration scores never contained.

## Does tuned BCI beat the selector?

| Domain | Model | Horizon | Diversity-optimal | Tuned BCI | div-opt width | BCI width | BCI separates above? |
|---|---|---|---|---|---|---|---|
| Recession | stacking-chain | Current | 93.85 | 89.23 | 108.7 | 20.9 | no |
| Recession | stacking-chain | 1M | 96.88 | 82.81 | 92.1 | 8.2 | no |
| Recession | stacking-chain | 3M | 95.16 | 83.87 | 109.1 | 13.2 | no |
| Recession | stacking-chain | 6M | 96.61 | 57.63 | 123.9 | 27.0 | no |
| Healthcare | ridge | 30-day | 92.47 | 86.99 | 25.3 | 20.9 | no |
| Healthcare | gradboost | 30-day | 93.84 | 88.36 | 25.7 | 21.8 | no |
| Climate | ridge | region-month | 99.59 | 91.36 | 17.8 | 10.7 | no |
| Climate | gradboost | region-month | 98.77 | 90.95 | 19.4 | 11.4 | no |

**0 of 8.** In no domain, under no model, does tuned BCI's Wilson lower bound
exceed the selector's coverage.

**But the efficiency comparison is genuinely favourable to BCI and should be
reported.** Tuned BCI reaches 90.95–91.36% in climate at roughly *half* the
selector's interval width (10.7–11.4 vs 17.8–19.4), and 86.99–88.36% in
healthcare at ~85% of the width. If a practitioner's target is "hit nominal as
tightly as possible" rather than "maximise coverage", tuned BCI is the better
choice in those two domains. The selector wins on coverage; BCI wins on
sharpness. That tradeoff is worth a sentence in §7.

## Honest caveats

- **The validation stream is synthetic in construction.** Pool points are stored
  as scores |pred − y|, not as (y, prediction) pairs, so the validation stream
  is reconstructed as those errors around a zero-centred prediction. This
  preserves the error distribution BCI must adapt to, but it is not a real
  held-out forecast stream, and a different reconstruction could select
  different parameters.
- **60 configurations is a coarse grid.** A finer or continuous search could
  find better settings; these numbers are a lower bound on tuned BCI's ability,
  not a ceiling.
- **Tuned coverage is not free.** Every tuned cell is wider than its untuned
  counterpart (1.4× to 5.9×). Tuning moved BCI along the coverage-width
  tradeoff, it did not eliminate it.
- **Recession 6M remains unsolved by BCI under any grid point tested.**
