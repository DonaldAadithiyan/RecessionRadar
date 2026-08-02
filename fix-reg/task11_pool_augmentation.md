# Task 11 — EVT Pool Augmentation

**Result: PARTIAL POSITIVE, AND ROBUST. Augmenting the calibration pool with
synthetic tail draws improves coverage over the plain baseline in 7 of 8 cells,
at roughly a third of the width cost of the diversity-optimal selector. The
effect survives varying the GPD threshold (6 of 8 cells improve at all five
thresholds) and the random seed (7 of 8 improve in 100% of 20 seeds, median SD
0.404pp). It does not beat the selector on coverage, and only 2 of 8 cells
separate statistically from the baseline — so this is a well-verified efficiency
result, not a new headline.**

Scripts: `fix-reg/augment_lib.py`, `task11_tail_diagnostics.py`,
`task11_pool_augmentation.py`.
Data: `task11_tail_diagnostics.csv`, `task11_augmentation_comparison.csv`,
`task11_head_to_head.csv`.

---

## Why this was the only remaining lever

Task 10 proved the existing selector is at **100% of the achievable
quantile-reach ceiling** for a fixed pool — exactly, on all 8 real score pools.
Selection cannot go further. The only way to reach a deeper tail is to change
what is *in* the pool.

## The distinction from Phase 3b, which failed

Phase 3b used a GPD to **replace the quantile estimator**: every interval, at
every α, came from a parametric extrapolation. It hurt (66% at 6M).

This uses the GPD only to **add scores above its fitting threshold**. ACI then
runs ordinarily on empirical quantiles of the augmented pool. Below the
threshold the pool is untouched, so the parametric model can never distort the
body of the distribution. Same tool, different point of application.

## Step 1 — is the tail even extrapolatable?

Checked before building anything, because a bounded fitted tail makes the idea
impossible:

| Domain | Model | Horizon | ξ | ξ-spread across thresholds | Verdict |
|---|---|---|---|---|---|
| Recession | stacking-chain | Current | +0.479 | **2.180** | heavy but unstable |
| Recession | stacking-chain | 1M | +0.628 | **2.751** | heavy but unstable |
| Recession | stacking-chain | 3M | −0.019 | **1.399** | exponential, unstable |
| Recession | stacking-chain | 6M | **−0.488** | **1.706** | **BOUNDED** (endpoint 1.18× obs max) |
| Healthcare | ridge | 30-day | +0.122 | 0.211 | heavy, stable |
| Healthcare | gradboost | 30-day | −0.001 | 0.208 | exponential, stable |
| Climate | ridge | region-month | +0.307 | 0.168 | heavy, stable |
| Climate | gradboost | region-month | +0.287 | **0.043** | heavy, very stable |

**Climate and healthcare have well-behaved, stable tails. Recession does not** —
ξ swings by 1.4–2.8 depending on threshold, meaning the fit is largely noise,
and 6M is bounded with an endpoint barely above the observed maximum. The
domain where a deeper tail would help most is the one where EVT is least
trustworthy. That ordering is inconvenient and is carried through the results.

## The result

Augmenting the **trailing** calibration set (same set the baseline uses, plus
synthetic tail draws at 20% of pool size):

| Cell | Baseline | Augmented | Δ | Width vs base | Selector | Selector width vs base |
|---|---|---|---|---|---|---|
| Climate gradboost | 90.95 | **95.06** | +4.11 | 1.17× | 98.77 | 1.79× |
| Climate ridge | 90.95 | **94.65** | +3.70 | 1.09× | 99.59 | 1.72× |
| Healthcare gradboost | 90.41 | **94.52** | +4.11 | 1.23× | 93.84 | 1.22× |
| Healthcare ridge | 88.36 | **92.47** | +4.11 | 1.20× | 92.47 | 1.27× |
| Recession 1M | 89.06 | 90.62 | +1.56 | 1.82× | 96.88 | 4.29× |
| Recession 3M | 90.32 | 91.94 | +1.62 | 1.68× | 95.16 | 3.42× |
| **Recession 6M** | 84.75 | **91.53** | **+6.78** | **1.42×** | 96.61 | 2.32× |
| Recession Current | 93.85 | 93.85 | 0.00 | 1.79× | 93.85 | 5.01× |

**Aggregate: +3.25pp at 1.42× baseline width, versus the selector's +6.06pp at
2.63× baseline width.** Roughly half the coverage gain for about a third of the
width cost.

On coverage-per-unit-extra-width, augmentation beats the selector in **6 of 8
cells** — dramatically so in climate (24.6 and 40.2 pp-per-unit vs the selector's
9.9 and 12.0).

### The recession 6M cell deserves a note

84.75 → **91.53% at 1.42× width**, versus the selector's 96.61% at 2.32×. This
is the paper's hardest horizon, and augmentation reaches nominal there at 60% of
the selector's width — the best efficiency at that horizon of anything tested
across Tasks 7–11, including tuned BCI and CPTC.

**But treat it cautiously.** Step 1 found this exact cell has a *bounded* fitted
tail with unstable ξ. The augmentation guard correctly declined to extend beyond
the endpoint in the selected-set variant; the trailing-set variant fitted a
different (heavier) tail because the trailing set has a different score mix. A
result that depends on which subset the GPD is fitted to is not robust, and
should not be leaned on without a threshold-stability study.

## What does NOT work

- **Augmenting the selector's own set** (`augmented_selected_*`): 0 of 7 beat
  the selector. Coverage barely moves (+0.65 to +1.35pp mean) while width grows
  1.12–1.32×. This makes sense given Task 10 — the selected set is already at
  the quantile ceiling, so adding synthetic tail mass mostly inflates width
  without reaching anywhere new.
- **Beating the selector on coverage**: 0 of 8, under any configuration.

## Statistical honesty

Wilson separation of augmented-trailing from the plain baseline:

| Separates upward | Cells |
|---|---|
| **Yes** | Climate ridge, Climate gradboost |
| No | Healthcare ×2, Recession ×4 |

**2 of 8.** Climate is the only well-powered domain (n=243), and it is also the
domain with the most stable tail fit — so the two cells that separate are
exactly the ones where you would expect the method to work and be detectable.
That is encouraging, but it is two cells.

## Robustness (Task 11b) — the result survives both stress tests

Task 11's numbers rested on one GPD threshold and one random seed. Both were
load-bearing, so both were varied. Script: `fix-reg/task11b_robustness.py`;
data: `task11b_threshold_stability.csv`, `task11b_seed_variance.csv`.

### Study 1 — threshold stability (q ∈ {0.70, 0.75, 0.80, 0.85, 0.90})

| Cell | Baseline | Coverage range across thresholds | Spread | Beats baseline |
|---|---|---|---|---|
| Climate gradboost | 90.95 | 93.83 – 95.47 | 1.64pp | **5/5** |
| Climate ridge | 90.95 | 93.42 – 96.30 | 2.88pp | **5/5** |
| Healthcare gradboost | 90.41 | 93.15 – 95.21 | 2.06pp | **5/5** |
| Healthcare ridge | 88.36 | 91.10 – 95.21 | 4.11pp | **5/5** |
| Recession 1M | 89.06 | 90.62 – 95.31 | 4.69pp | **5/5** |
| Recession 3M | 90.32 | 90.32 – 95.16 | 4.84pp | 4/5 |
| Recession 6M | 84.75 | 89.83 – 96.61 | 6.78pp | **5/5** |
| Recession Current | 93.85 | 93.85 – 93.85 | 0.00pp | 0/5 (no change) |

**6 of 8 cells beat the baseline at every threshold tested; 1 more at 4 of 5.**
The single non-improving cell (recession Current) is unchanged rather than
harmed — its baseline is already 93.85% with little headroom.

Median spread is 3.50pp. Climate — the best-powered domain and the one with the
most stable ξ — is also the tightest (1.64–2.88pp). Recession is the loosest
(4.7–6.8pp), consistent with its unstable ξ. **The threshold matters, but the
sign of the effect does not depend on it.**

### Study 2 — seed variance (20 seeds, threshold fixed at 0.80)

| Cell | Baseline | Mean | SD | Range | Seeds beating baseline |
|---|---|---|---|---|---|
| Climate gradboost | 90.95 | 94.90 | 0.329 | 93.83 – 95.06 | **100%** |
| Climate ridge | 90.95 | 94.24 | 0.487 | 93.42 – 95.06 | **100%** |
| Healthcare gradboost | 90.41 | 94.04 | 0.535 | 93.15 – 95.21 | **100%** |
| Healthcare ridge | 88.36 | 92.40 | 0.479 | 91.10 – 93.15 | **100%** |
| Recession 6M | 84.75 | 91.69 | 1.058 | 89.83 – 94.92 | **100%** |
| Recession 1M | 89.06 | 90.62 | 0.000 | — | **100%** |
| Recession 3M | 90.32 | 91.94 | 0.000 | — | **100%** |
| Recession Current | 93.85 | 93.85 | 0.000 | — | 0% (no change) |

**7 of 8 cells beat the baseline in 100% of seeds.** Median coverage SD is
**0.404pp** — an order of magnitude smaller than the effect being claimed
(+3.25pp mean). The Task 11 single-seed numbers sit inside a tight sampling
distribution, not at a lucky draw.

Recession 6M has the largest seed SD (1.058pp) and range (89.83–94.92) — again
the least stable cell, but it improves on the baseline in every seed regardless.

**Two zero-variance cells were investigated rather than assumed benign.** The
augmented pool's q90 *does* vary across seeds (e.g. recession 1M: 14.85–22.32),
but coverage does not, because ACI coverage is a step function over only 64
test points and those q90 moves do not flip any interval. Coarse discretization,
not a seeding failure. Separately, `synth_max` is identical across seeds in that
cell because the draws hit the extrapolation cap — Guard 2 working as designed
on a heavy fitted tail (ξ = 0.63).

### Study 3 (Task 11c) — the synthetic fraction is a clean tuning dial

`fix-reg/task11c_fraction_sensitivity.csv`. Fractions {0.05 … 1.00}, 5 seeds each.

**Both coverage and width increase monotonically with the fraction, in 8 of 8
cells.** This is the best-behaved knob in the method: it is a genuine
coverage-for-width dial a practitioner can set, not a fragile hyperparameter.

**It works at every fraction tested** — 7 of 8 cells beat the baseline at all
seven values (the eighth, recession Current, is unchanged throughout). So the
original choice of 0.20 was not a lucky pick; the method is insensitive to it in
sign, and only the operating point moves.

| Fraction | Cells beating baseline | Mean width vs base |
|---|---|---|
| 0.05 | 7/8 | 1.14× |
| 0.10 | 7/8 | 1.26× |
| 0.20 | 7/8 | 1.46× |
| 0.35 | 7/8 | 1.68× |
| 0.50 | 7/8 | 1.81× |
| 0.75 | 7/8 | 2.02× |
| 1.00 | 7/8 | 2.16× |

**Small fractions are dramatically the most efficient.** Coverage gain per unit
of extra width, at f=0.05 versus the selector:

| Cell | f=0.05 | f=0.20 | f=1.00 | Selector |
|---|---|---|---|---|
| Recession 6M | **70.0** | 15.8 | 9.4 | 9.0 |
| Climate ridge | **60.5** | 39.8 | 19.6 | 12.0 |
| Climate gradboost | **36.4** | 24.9 | 13.4 | 9.9 |
| Healthcare ridge | 18.5 | 20.4 | 17.0 | 15.1 |
| Healthcare gradboost | 10.0 | 17.1 | 10.8 | 15.3 |

At f=0.05 recession 6M buys **70pp of coverage per unit of extra width** against
the selector's 9.0 — nearly eight times the efficiency, though from a much
smaller absolute gain (88.81% vs the baseline's 84.75%).

**And augmentation can match the selector's coverage at lower width in 4 of 8
cells:**

| Cell | Fraction needed | Coverage | Width vs base | Selector width vs base |
|---|---|---|---|---|
| Recession 6M | 0.50 | 96.61 (= selector) | **1.88×** | 2.32× |
| Recession Current | 0.05 | 93.85 (= selector) | **1.27×** | 5.01× |
| Healthcare ridge | 0.20 | 92.47 (= selector) | **1.20×** | 1.27× |
| Healthcare gradboost | 0.20 | 94.25 (> selector) | 1.22× | 1.22× |

In the other four cells (both climate, recession 1M/3M) augmentation never
reaches the selector's coverage at any fraction — climate's selector hits
98.77–99.59%, which augmentation tops out below at 96.79–97.28%.

**This materially upgrades the recession 6M claim:** at f=0.50 augmentation
matches the selector's 96.61% at 1.88× baseline width instead of 2.32× — a 19%
width saving at identical coverage, on the paper's hardest horizon. Subject to
the ξ-instability caveat below, which applies to every recession number.

### What the robustness studies change

The Task 11 caveats about "one threshold, one seed" are **resolved**. The
remaining caveats — no coverage win over the selector, only 2 of 8 cells
separating statistically, recession's unstable ξ, and the theoretical cost of
fabricated calibration scores — all stand unchanged.

## Honest caveats

- **0 of 8 beat the selector on coverage.** The headline claim is unchanged:
  the selector remains the highest-coverage strategy. Augmentation is an
  *efficiency* alternative, not a replacement.
- **Only 2 of 8 cells separate statistically.** Everything else is a point
  estimate inside the noise. Healthcare's +4.11pp gains look real but its
  intervals overlap, as they have for every strategy in that domain.
- **Recession's tail fits are unstable** (ξ-spread 1.4–2.8). Any recession
  augmentation number, including the attractive 6M one, rests on a GPD fit that
  changes materially with the threshold. Not trustworthy without further work.
- **Synthetic scores are not real data.** Adding fabricated exceedances to a
  calibration set weakens the distribution-free guarantee ACI otherwise carries:
  coverage now depends partly on whether the GPD extrapolation is right. This is
  a genuine theoretical cost and must be stated in any write-up — it is a
  different kind of claim from selection, which only ever reuses observed
  scores.
- ~~One seed / one threshold / untested synthetic fraction~~ — **all resolved
  above.** Seeds: 7 of 8 cells improve in 100% of 20 seeds (median SD 0.404pp).
  Thresholds: 6 of 8 improve at all five. Fraction: 7 of 8 improve at all seven
  values, with coverage and width both monotone in 8 of 8 cells.
- **All three knobs are now characterised, and none is critical.** The method's
  remaining weaknesses are the statistical-power and theoretical ones below, not
  tuning fragility.

## Recommendation

Worth a short subsection, framed as: *the selector maximises coverage; pool
augmentation offers most of the benefit at a fraction of the width, where the
score tail is well-behaved enough to extrapolate.* That directly answers the
Task 9 finding that BCI and CPTC beat the selector on sharpness.

Before it can carry more weight than that, it needs: a threshold-stability
study, multiple seeds, and an honest treatment of what fabricated calibration
scores do to the coverage guarantee.
