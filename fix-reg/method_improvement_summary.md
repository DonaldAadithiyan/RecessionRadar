# Method Improvement — Summary of Tasks 10–11c

Three attempts to improve the diversity-optimal selector, prompted by the Task 9
finding that tuned BCI and CPTC beat it on **sharpness** while it wins on
coverage. One failed informatively, one worked partially, and together they
bound what is achievable.

| Task | Attempt | Outcome |
|---|---|---|
| **10** | Quantile-targeted selection (replace the p95−p5 objective) | **Negative** — the existing selector is already at 100% of the achievable ceiling |
| **11** | EVT pool augmentation (extend the pool, don't reselect) | **Partial positive** — +3.25pp at 1.42× width vs the selector's +6.06pp at 2.63× |
| **11b** | Threshold + seed robustness | **Survives both** — 6/8 cells improve at all thresholds, 7/8 in 100% of seeds |
| **11c** | Synthetic-fraction sensitivity | **Clean tuning dial** — coverage and width both monotone in 8/8 cells; works at every fraction tested |
| **12A** | Does optimality survive small N? | **Yes, and structurally** — 56/56 configs at ratio 1.000000, N from 254 down to 20; holds for any operating quantile above the median |
| **12B** | α-drift re-selection | **Falsified** — 0/8, identical to static; re-selected sets differ in membership but give identical operative quantiles |
| **13** | Temporal generalization (3 historical cutoffs) | **Holds — 12/12 cells.** Best-powered new cutoff (2006, 16 rare months) shows a *larger* gap than the published split |
| **14** | Compute cost across 9 strategies | Selector is **1.0× baseline per step**, 0.06ms setup. EVT-tail is 110–199×; CPTC's state model is 190ms one-off |
| **15** | Healthcare power vs headroom | **Low-headroom, not underpowered.** Baseline is 88–91% under every admissible target; separations appear only once the target is degraded past the noise floor |

---

## The through-line

Task 10 asked: *can a smarter selection rule do better?* Answer: **no, and
provably so.** ACI consumes only the (1−α) quantile of the calibration scores,
and the existing selector already attains the maximum achievable value of that
quantile for a fixed pool — exactly, ratio 1.0000, on all 8 real score pools at
every operative quantile level. Two selectors sharing only 31% of their indices
produce identical operative quantiles, because any upper-tail-loading subset
lands on the same extreme pool scores.

That closed off selection as a lever and pointed at the only remaining one:
**change what is in the pool.**

Task 11 did that via EVT augmentation, and it works — but as an *efficiency*
gain, not a coverage gain. It recovers about half the selector's coverage
improvement for about a third of its width cost, which is precisely the axis
where BCI and CPTC were winning.

## What this means for the paper's claims

**Strengthened:** Phase 2's selector now has an optimality argument for the
quantity Phase 5 proved is operative, not merely for the p95−p5 proxy it was
designed around. Two unrelated objectives, same optimum.

**And that argument is now unhedged (Task 12A).** It holds at every N from 254
down to 20 — 56 of 56 configurations at ratio exactly 1.000000 — with a
structural reason: the alternating rule supplies ~N/2 extreme-tail scores while
the operative quantile depends on only the top 7–13%, so it is oversupplied ~4×
regardless of N. The claim is falsifiable (it would break at an operating
quantile below the median) but no calibration regime in this paper approaches
that.

**Newly explained:** the selector's 1.2–2.6× width cost is **intrinsic, not a
design flaw**. Reaching the required tail quantile requires those extreme
scores; the N-largest set attains the same q90 with nearly double the median
(uniformly wide). The existing rule already takes the cheapest route there.

**New option, honestly bounded:** pool augmentation as a sharpness-oriented
alternative, for practitioners targeting "nominal, as tight as possible" rather
than "maximum coverage". The synthetic fraction is a monotone dial (8/8 cells)
along that tradeoff, so the operating point is a deliberate choice rather than a
fitted hyperparameter.

**The sharpest version of the claim** (Task 11c): in 4 of 8 cells augmentation
*matches the selector's coverage at lower width* — including recession 6M, where
it reaches the selector's 96.61% at 1.88× baseline width instead of 2.32×, a 19%
width saving at identical coverage on the paper's hardest horizon. In the other
4 cells (both climate, recession 1M/3M) it never reaches the selector's
coverage at any fraction.

## The honest limits

- Augmentation **never beats the selector on coverage** (0 of 8).
- Only **2 of 8 cells separate statistically** from the baseline, both climate —
  the best-powered domain and the one with the most stable tail fit.
- **Recession's GPD fits are unreliable** (ξ-spread 1.4–2.8 across thresholds;
  6M formally bounded). The domain where a deeper tail would help most is where
  EVT is least trustworthy.
- **Fabricated calibration scores weaken the distribution-free guarantee.**
  Coverage now depends partly on the extrapolation being correct. This is a
  theoretical cost that no amount of resampling addresses, and it is the real
  remaining objection — it should be argued explicitly rather than buried under
  robustness tables.

## Files

- `task10_quantile_selection.md` — the negative result and the ceiling proof
- `task11_pool_augmentation.md` — augmentation, with Task 11b robustness folded in
- `task11c_fraction_sensitivity.csv` — the synthetic-fraction sweep
- Code: `selector_lib.py`, `augment_lib.py`,
  `task10_*.py`, `task11*.py`
