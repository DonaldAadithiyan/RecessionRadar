# Task 7 — Cross-Domain Baseline Comparison

Closes the limitation named in §7 of the paper: the standard-toolbox comparison
was previously run on the recession testbed only. All eight strategies now run
on all three domains, out-of-fold, with confidence intervals on every number.

**Headline: the diversity-optimal selector is the only strategy that separates
upward from the pooled/trailing baseline anywhere, and it does so in two of the
three domains. No newer method (2024–2025) beats it, and three of the four
newer methods are indistinguishable from or worse than plain ACI here.**

Scripts: `fix-reg/baselines_lib.py`, `fix-reg/task7_baseline_horse_race.py`.
Data: `task7_baselines_{recession,healthcare,climate}.csv`.
Figure: `figures/methods_coverage_comparison_alldomains.{pdf,png}`.

---

## Bug check first: Phase 3 reproduces exactly

Per the spec, the recession rows were recomputed and checked against the
published Phase 3 figures **before** any new numbers were reported. The harness
aborts rather than proceeding if they disagree.

**All 16 reference cells (4 strategies × 4 horizons) reproduce to the stated
precision.** Pooled/trailing, Mondrian, PID-conformal and EVT-tail all match
`phase3a_mondrian.csv` / `phase3bc_extra_baselines.csv` exactly. PID and EVT are
byte-for-byte ports of the Phase 3 code rather than reimplementations, precisely
so this check is meaningful.

---

## Domain 1 — Recession (out-of-fold, stacking-chain)

Coverage % with 95% Wilson interval; width multiple relative to pooled/trailing.

| Strategy | Current | 1M | 3M | 6M | width× at 6M |
|---|---|---|---|---|---|
| Pooled/trailing ACI | 93.85 | 89.06 | 90.32 | 84.75 | 1.00 |
| Mondrian | 92.31 | 87.50 | 85.48 | 83.05 | 0.78 |
| PID-conformal | 93.85 | 90.62 | 90.32 | 86.44 | 0.96 |
| EVT-tail | 93.85 | 87.50 | 90.32 | 83.05 | 1.04 |
| DtACI (2024) | 93.85 | 89.06 | 91.94 | 89.83 | 1.00 |
| AcMCP (2024) | 90.77 | 85.94 | 85.48 | **64.41** | 0.55 |
| Bellman CI (2024) | 80.00 | 79.69 | 61.29 | **49.15** | 0.09 |
| **Diversity-optimal (ours)** | 93.85 | **96.88** | 95.16 | **96.61** | 2.32 |

**Verdict.** Only diversity-optimal separates upward from the baseline (1M and
6M: Wilson lower bound above the baseline point estimate). DtACI is the best of
the newer methods — it lifts 6M from 84.75% to 89.83% *at essentially no width
cost* (1.00×), which is a genuinely attractive profile — but the gain does not
clear the confidence interval, so it must be reported as suggestive only.

**AcMCP collapses at 6M** (64.41%, and with *narrower* intervals than the
baseline, 0.55×). Its autocorrelation term is driving width down exactly where
the shift demands it go up. Reported as a clean negative result.

## Domain 2 — Healthcare (30-day readmission, out-of-fold)

| Strategy | ridge | gradboost | width× (gradboost) |
|---|---|---|---|
| Pooled/trailing ACI | 88.36 | 90.41 | 1.00 |
| Mondrian | 91.78 | 91.78 | 1.32 |
| PID-conformal | 88.36 | 90.41 | 0.96 |
| EVT-tail | 89.04 | 90.41 | 1.03 |
| DtACI (2024) | 88.36 | 89.04 | 0.98 |
| AcMCP (2024) | 80.82 | 89.73 | 1.95 |
| Bellman CI (2024) | 69.18 | 66.44 | 0.56 |
| **Diversity-optimal (ours)** | **92.47** | **93.84** | 1.22 |

**Verdict.** No strategy separates upward here — the baseline already sits at
or near nominal (88.4–90.4%), so there is little headroom and every interval
overlaps. Diversity-optimal has the highest point estimate under both models,
but this is **not** a claim of a win: the CIs overlap the baseline. AcMCP
separates *downward* under ridge (80.82%).

## Domain 3 — Climate (storm intensity, out-of-fold)

| Strategy | ridge | gradboost | width× (gradboost) |
|---|---|---|---|
| Pooled/trailing ACI | 90.95 | 90.95 | 1.00 |
| Mondrian | **95.88** | **94.65** | 2.01 |
| PID-conformal | 89.71 | 88.89 | 0.95 |
| EVT-tail | 90.53 | 91.36 | 0.99 |
| DtACI (2024) | 90.12 | 90.95 | 0.98 |
| AcMCP (2024) | 88.89 | 88.07 | 1.47 |
| Bellman CI (2024) | 78.19 | 79.42 | 0.73 |
| **Diversity-optimal (ours)** | **99.59** | **98.77** | 1.79 |

**Verdict.** Diversity-optimal and Mondrian both separate upward, under both
models. Note the coverage is now *above* nominal at 98.8–99.6% — overcoverage
at ~1.8× width, which is a cost, not a pure win.

### Mondrian reverses sign across domains — and the reason is instructive

Mondrian **backfires on recession** (6M: 83.05 vs 84.75 baseline; Phase 3 saw
the same, 61.02 vs 67.80 in-sample) but **helps on climate** (+3.7 to +4.9pp,
CI-separated). The difference is regime balance in the test window:

- Recession test window: 63 expansion months vs **2** rare months. The
  class-conditional pool for the rare regime is starved, so Mondrian stops
  borrowing the wide rare-event scores that were carrying coverage.
- Climate test window: **120 storm-season vs 123 off-season** region-months —
  balanced, so both conditional pools are well populated.

This is direct evidence that Mondrian's documented failure in this paper is a
*degeneracy of the recession test window*, not a general property of
class-conditional calibration. That nuance was not visible from the
single-domain comparison and should go in the paper.

---

## Design decisions that a reader must be able to audit

**Mondrian regime labels outside recession** (no natural expansion/recession
split exists):

- **Healthcare:** top-tercile predicted readmission risk vs the rest — a
  risk-based partition, chosen as the analog of expansion/recession being a
  risk-based partition of the economy. This is a design choice, not a given.
- **Climate:** named-storm-season (June–November) vs off-season. Regime counts
  checked for degeneracy before running: **120 vs 123** test region-months, so
  non-degenerate, unlike the recession window's 63/2.

**Block length for the bootstrap:** 12 months for recession and climate
(monthly data with annual structure); **block = 1 for healthcare**, because its
units are patient cohorts, which are exchangeable rather than a time series —
a moving-block bootstrap would impose serial structure that does not exist.

**DtACI determinism:** the paper samples α_t from the expert distribution; we
take the probability-weighted mean instead. On test streams of 59–243 points,
Monte-Carlo sampling noise would be comparable to the effects being measured.
This makes our DtACI slightly more stable than the sampled version.

---

## The two hardest methods, reported honestly

### Bellman Conformal Inference — implemented, and it undercovers

Implemented from the authors' released code
(`github.com/ZitongYang/bellman-conformal-inference`), reimplemented rather than
imported because their package couples the DP to their own dataloader and pins
numpy 1.23, which conflicts with this project's stack.

**A first version of this was wrong and is worth recording.** Written with a
fixed λ, it returned 0–4% coverage: the interval-length cost (range ~49 in raw
score units) swamped the coverage-violation cost (range ~10), so the dynamic
program always chose the narrowest interval. Inspecting the authors' configs
showed why — **λ is not a hyperparameter in their method at all**. It is a
control variable updated online, `λ ← λ − γ(α₀ − err)`, saturating to α=0 or
α=1 at its bounds. The corrected implementation uses their released defaults
(`λ_init=5, λ_max=500, γ=0.8, T=3`) with **no tuning on our data**.

BCI undercovers in every domain (49–80%) with very narrow intervals
(0.09–0.75× baseline). This is a coherent length-penalised profile, not a bug:
BCI explicitly optimises average interval length subject to a coverage
constraint, and on these short, shift-heavy streams it trades away coverage. It
separates downward from the baseline in 8 of 8 comparisons.

**Caveat stated plainly:** because a hand-tuned λ can move BCI's coverage
anywhere between 3% and 94% (verified on synthetic data), any BCI number is
only meaningful together with its parameter provenance. Ours are the authors'
published defaults, untuned. We do not claim BCI is a weak method in general —
only that, out of the box, it does not address this paper's coverage problem.

### CPTC — not attempted, and why

**Not run.** CPTC's algorithm requires `z_prob`, a per-timestep state-probability
matrix produced by a *separately trained* REDSDS switching-dynamics model; the
authors ship these as precomputed `.npz` files for their six datasets. Producing
them for recession/healthcare/climate means training an additional latent-state
model per domain.

That is barred by this task's own guardrail ("do not retrain any of the
underlying point-prediction models — this is a calibration-layer comparison
only") and would make CPTC non-comparable with the other seven strategies, all
of which consume the same fixed nonconformity scores. Approximating `z_prob`
with, say, a hand-rolled regime indicator would be a different algorithm wearing
CPTC's name — exactly what the spec says not to ship.

Recorded as a clean "not attempted, here's why" rather than a best-effort
approximation.

---

## Honest caveats

- **No newer method beat ours, but the field is not exhausted.** Four newer
  methods were tested; CPTC (the most directly change-point-aware, and
  plausibly the strongest candidate) was not. The claim is "none of the
  baselines we ran beats diversity-optimal", not "nothing beats it".
- **Diversity-optimal buys coverage with width** (1.2–2.3× baseline; up to
  4.3× at recession 1M). Consistent with Phase 2's finding; it is not a free
  lunch and must never be reported on coverage alone.
- **Healthcare separates nothing.** With the baseline already at nominal, that
  domain discriminates poorly between strategies. Its value here is confirming
  the mechanism transfers, not ranking methods.
- **Climate overcovers** (98.8–99.6%) under diversity-optimal. Above-nominal
  coverage at ~1.8× width is a real cost, not a clean victory.
- **n is small at recession 6M** (59 scored months); intervals are ±10pp wide.
  All recession verdicts inherit that imprecision.
- **Ranking by point estimate is not supported anywhere in this table.** Only
  the CI-separation column licenses a "beats" claim; 12 of 16 non-baseline
  recession comparisons separate from nothing at all.
