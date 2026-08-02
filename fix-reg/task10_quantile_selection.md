# Task 10 — Quantile-Targeted Calibration Selection

**Result: NEGATIVE. The idea does not work, and the reason it does not work is
itself a useful result — the existing selector is provably already optimal on
the quantity that governs coverage.**

Scripts: `fix-reg/selector_lib.py`, `task10_alpha_trajectories.py`,
`task10_quantile_selector.py`.
Data: `task10_alpha_ranges.csv`, `task10_selector_comparison.csv`,
`task10_head_to_head.csv`.

---

## The hypothesis

The existing selector maximises support width (p95 − p5). But ACI never
consumes p95 − p5 — at each step it consumes exactly one number, the empirical
(1 − α_t) quantile of the calibration scores. Widening the *lower* tail should
therefore contribute nothing to coverage while inflating width.

Predicted consequence: an objective targeting quantile reach over the α window
ACI actually visits, subject to a width budget, should hold coverage at
materially lower width — attacking the one axis where tuned BCI and CPTC
currently beat the selector (Task 9).

## Step 1 — the α window, measured rather than assumed

| Domain | Model | Horizon | α range visited | Median α |
|---|---|---|---|---|
| Recession | stacking-chain | Current | [0.0830, 0.1120] | 0.0965 |
| Recession | stacking-chain | 1M | [0.0730, 0.1000] | 0.0840 |
| Recession | stacking-chain | 3M | [0.0825, 0.1010] | 0.0915 |
| Recession | stacking-chain | 6M | [0.0760, 0.1000] | 0.0845 |
| Healthcare | ridge | 30-day | [0.0780, 0.1210] | 0.1043 |
| Healthcare | gradboost | 30-day | [0.0930, 0.1210] | 0.1100 |
| Climate | ridge | region-month | [0.0960, 0.1230] | 0.1085 |
| Climate | gradboost | region-month | [0.1000, 0.1320] | 0.1190 |

**Overall: α ∈ [0.073, 0.132]**, so the operative calibration quantile is always
in **[0.868, 0.927]**. This is a useful measurement in its own right — ACI stays
in a narrow band around nominal under these calibration sets, and the p5 end of
the support-width objective is provably irrelevant to it.

## Step 2 — the result

Eight cells (3 domains × models × horizons), four new arms each (unconstrained,
plus width budgets κ ∈ {0.75, 1.0, 1.5}):

- **0 of 8 strict wins** (≥ coverage and < width) for every arm.
- **Median width vs the existing selector: 1.000×.**
- Mean coverage change −0.41pp.
- In **6 of 8 cells the results are byte-identical** to the existing selector.
  In the two climate cells the new selector is slightly *worse* on both axes
  (−1.65pp coverage at 0.90–0.92× width).

Identical outputs are the diagnostic signal here, not a coding artifact.

## Why it fails — the useful part

The two selectors share only **31% of their selected indices**, yet produce
**identical quantiles at every operative level**:

| Quantile | Support-width selector | Quantile-targeted |
|---|---|---|
| q(0.868) | 6.1339 | 6.1339 |
| q(0.900) | 6.8212 | 6.8212 |
| q(0.927) | 7.5038 | 7.5038 |

The mechanism: at N=254 drawn from a ~1200-score pool, *any* selector that
loads the upper tail places the selected set's 87th–93rd percentile on the same
extreme pool scores. The lower-tail choices differ, but they fall below the
operative quantile and never enter the interval ACI builds.

Checking against the theoretical ceiling — the q90 of the N **largest** scores,
which no N-subset can exceed — settles it:

```
q90 of the N largest scores (max achievable) : 6.8212
q90 achieved by the support-width selector   : 6.8212
=> the existing selector is at 100.0% of the ceiling
```

**This is not a synthetic-data artifact — it holds exactly on every real domain
score pool**, at all three operative quantiles (ratio = achieved / ceiling):

| Cell | pool size | q=0.868 | q=0.900 | q=0.927 |
|---|---|---|---|---|
| Recession Current | 545 | 1.0000 | 1.0000 | 1.0000 |
| Recession 1M | 545 | 1.0000 | 1.0000 | 1.0000 |
| Recession 3M | 545 | 1.0000 | 1.0000 | 1.0000 |
| Recession 6M | 545 | 1.0000 | 1.0000 | 1.0000 |
| Healthcare ridge | 438 | 1.0000 | 1.0000 | 1.0000 |
| Healthcare gradboost | 438 | 1.0000 | 1.0000 | 1.0000 |
| Climate ridge | 1148 | 1.0000 | 1.0000 | 1.0000 |
| Climate gradboost | 1148 | 1.0000 | 1.0000 | 1.0000 |

Eight cells, three quantiles each, all exactly at the ceiling.

**The existing selector is already optimal on the quantity Phase 5 identified as
operative.** It reaches that optimum incidentally — its extreme-tail alternation
happens to capture the same scores — but it reaches it exactly. There is no
headroom for a smarter selection rule to exploit.

## What this means for the paper

This is a **strengthening result, not a failed experiment**, and it should be
reported as such:

1. **Phase 2's selector now has an optimality argument, not just an empirical
   one.** It was previously justified as "greedy is provably optimal for
   p95 − p5". The stronger statement is: it is also optimal for weighted
   quantile reach over the α window ACI actually visits — the quantity Phase 5
   proved governs the coverage deficit. Two different objectives, same optimum.

2. **It closes off a whole class of future work.** "Use a smarter calibration
   selection objective" is now a tested dead end, with the ceiling computation
   to show why. That is worth one paragraph in §7 to save the next reader the
   effort.

3. **The width cost is not a selection failure — it is intrinsic.** The
   selector's 1.2–2.6× width penalty cannot be engineered away by reselecting,
   because reaching the required tail quantile *requires* those extreme scores.
   The N-largest set achieves the same q90 with a far higher median (3.29 vs
   1.79), i.e. it is uniformly wide; the existing selector already takes the
   cheapest route to that quantile. **Any real width improvement must come from
   somewhere other than subset selection** — which redirects the remaining ideas
   (adaptive re-selection as α drifts; synthetic tail augmentation) toward
   changing *what is in the pool* rather than *which pool members are chosen*.

## Honest caveats

- ~~The negative result is specific to this N/pool regime… not tested.~~
  **RESOLVED by Task 12A** (`task12_boundary_tests.md`): the ceiling is attained
  at ratio exactly 1.000000 in 56 of 56 configurations, with N swept from 254
  down to 20 across all 8 cells. There is also a structural reason — the
  alternating rule supplies ~N/2 extreme-tail scores while the operative
  quantile needs only the top (1−q)·N ≈ 7–13%, so it is oversupplied ~4× at
  every N. Optimality could only break at an operating quantile below the
  median (q < 0.5), far outside the measured ACI window.
- **The width budget never bound.** Because the objective saturates in `f` (the
  share drawn from the upper tail), all κ values selected the same subset. A
  budget that actually bit would have forced a narrower, lower-reach set — i.e.
  strictly worse coverage, which is not the tradeoff being sought.
- **The two climate cells got slightly worse**, not merely equal. The
  even-stride construction preserves less of the mid-distribution mass than the
  alternating rule, which marginally lowers coverage where the baseline is
  already over-nominal. A minor effect, reported for completeness.
- **This tests one objective.** A fundamentally different selection criterion
  (e.g. one optimising conditional coverage across regimes rather than marginal
  quantile reach) is not ruled out by this result.
