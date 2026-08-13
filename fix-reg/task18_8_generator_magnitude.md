# Task 18 Item 8 — Synthetic Generator Magnitude Adjustment

**Verdict: NEITHER modification helps. Both make the magnitude gap worse
(mean |error| 0.230 and 0.249 against the baseline's 0.209). Per the spec this
is a valid and informative outcome: the healthcare/climate over-prediction is
not a noise-structure problem, so it reflects something the generator's basic
structure cannot capture.**

The spec was explicit that this must not become a parameter search. One
structural change each, parameters set from a stated rationale before running,
run once, reported as-is.

Script: `fix-reg/task18_8_generator_magnitude.py`.
Data: `task18_8_generator_magnitude.csv`.

---

## The two modifications, and why those parameters

Both build on Task 17 Item 4's correlated latent-severity generator.

**Mod A — heteroskedastic noise.** `S_t = |ε_t·σ(Z_t) + Δ_S·Z_t|` with
`σ(Z) = exp(0.5·Z)`. Rationale stated in advance: real forecast errors plausibly
become *noisier*, not merely larger, in severe periods. The coefficient 0.5 is
half the unit scale of Z — a modest, non-tuned choice giving a ~1.65× noise-scale
ratio per unit of severity.

**Mod B — state-dependent coupling.** `a_t = a·(1 + 0.5·sin(2πt/120))`.
Rationale: the strength of the link between severity and event labelling is
itself regime-dependent (recording standards, policy thresholds and diagnostic
criteria drift). The 120-month period is the business-cycle scale; the 0.5
amplitude keeps `a_t` positive throughout.

## Result

Predicted gap vs actual, by variant:

| Domain | Actual | Baseline | Heteroskedastic | State-dependent |
|---|---|---|---|---|
| Recession 3M | +0.402 | +0.464 (1.15×) | **+0.402 (1.00×)** | +0.479 (1.19×) |
| Recession 6M | +0.442 | +0.341 (0.77×) | **+0.471 (1.07×)** | +0.319 (0.72×) |
| Healthcare | +0.122 | +0.439 (3.6×) | +0.518 (**4.25×**) | +0.495 (4.06×) |
| Climate | +0.062 | +0.419 (6.8×) | +0.558 (**9.0×**) | +0.486 (7.8×) |

| Variant | Mean abs error | Median ratio | Signs correct |
|---|---|---|---|
| **Baseline** | **0.209** | 2.38× | 4/4 |
| Heteroskedastic | 0.230 | 2.66× | 4/4 |
| State-dependent | 0.249 | 2.62× | 4/4 |

**Modifications improving on the baseline: none.**

## The result is more interesting than a flat null

Heteroskedastic noise **nearly perfectly matches recession** (3M: +0.402 against
an actual +0.402; 6M: +0.471 vs +0.442) while making healthcare and climate
**worse** (4.25× and 9.0× over-prediction). It does not fail uniformly — it
improves the domain it was implicitly modelled on and degrades the others.

That points at the real explanation. The over-prediction is not about noise
structure; it is that **healthcare and climate have genuinely weaker
diversity-vs-rare-count gaps than any variant of this generator produces**
(+0.122 and +0.062 measured, against +0.32 to +0.56 predicted by every variant).
Both are domains where Task 16B found rare-count retains real independent
predictive power and Task 15 found healthcare is low-headroom. A generator built
around a single latent severity driving everything will always make diversity
dominate strongly, because by construction there is nothing else going on.

**The honest conclusion:** the magnitude gap reflects domain structure the
generator omits — competing predictors, near-nominal baselines, weak headroom —
not a missing noise term. Closing it would require modelling those features, and
doing so by fitting to the observed gaps is exactly the overfitting Task 16C
diagnosed and the guardrail forbids.

## What the paper should say

Keep Task 17 Item 4's framing unchanged: the correlated generator corroborates
the mechanism **qualitatively** (4/4 correct signs) and should not be cited for
effect sizes. Add one sentence that two structural extensions were tried and
neither narrowed the magnitude gap, so the qualitative limitation is
characterised rather than merely acknowledged.

## Honest caveats

- **One parameter setting per modification.** By design. A sweep might find a
  setting that narrows the gap, but per the guardrail that would be the search
  this item exists to avoid.
- **The two modifications were not combined.** Whether heteroskedastic noise plus
  state-dependent coupling behaves differently from either alone was not tested.
- **One seed per cell**, inherited from Task 17 Item 4. The recession match under
  heteroskedastic noise (|err| = 0.000 at 3M) is close enough to be partly luck
  at this sample size and should not be read as validation.
- **Only the four real-domain coordinates were evaluated**, not the full phase
  plane, since the question was specifically about magnitude at those points.
