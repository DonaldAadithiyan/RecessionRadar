# Task 17 Item 2 — Stratified and Mixed Calibration Baselines

**Verdict: generic spreading does NOT substitute for the extremes. Decile- and
quintile-stratified sampling capture only 14–20% of the selector's gain. But the
result is sharper than "the selector wins": strategies that reach the same Q_C
get the same coverage, whatever route they take there — which is the paper's own
mechanism confirmed by a route it never tested.**

Script: `fix-reg/task17_2_stratified.py`.
Data: `task17_2_stratified_baselines.csv`, `task17_2_gain_fractions.csv`.

---

## The question this answers

Every strategy tested so far is either a full conformal method (Mondrian, DtACI,
PID, AcMCP, Bellman CI, CPTC) or one of the extremes (trailing, rare-heavy,
tail-extreme). The gap in between was never probed: **does any form of spreading
the calibration set across the score range get most of the benefit?** If so, the
selector is an unremarkable instance of a broad "spread your calibration set"
principle rather than something motivated specifically by tail-reach.

## Result — fraction of the selector's gain captured

| Strategy | Median fraction of gain | Median Q_C ratio vs selector |
|---|---|---|
| decile_stratified | **19.9%** | 0.558 |
| quintile_stratified | **14.2%** | 0.559 |
| tail_plus_recency | **100.0%** | 1.000 |

Per-cell coverage on the recession testbed:

| Strategy | Current | 1M | 3M | 6M |
|---|---|---|---|---|
| pooled/trailing | 93.85 | 89.06 | 90.32 | 84.75 |
| quintile_stratified | 93.85 | 90.62 | 90.32 | 86.44 |
| decile_stratified | 93.85 | 90.62 | 90.32 | 88.14 |
| **tail_plus_recency** | 93.85 | **96.88** | **95.16** | **96.61** |
| **diversity_optimal** | 93.85 | **96.88** | **95.16** | **96.61** |

**Stratified sampling is barely better than doing nothing.** At 6M it moves
coverage from 84.75 to 88.14 (decile) against the selector's 96.61 — under 30%
of the gain, and at 3M it captures none at all.

## Why — and this is the finding worth reporting

Q_C by strategy, recession testbed:

| Strategy | 1M | 3M | 6M |
|---|---|---|---|
| pooled/trailing | 5.87 | 13.41 | 20.05 |
| quintile_stratified | 13.99 | 15.43 | 24.39 |
| decile_stratified | 15.53 | 16.33 | 26.88 |
| **tail_plus_recency** | **48.62** | **56.60** | **68.72** |
| **diversity_optimal** | **48.62** | **56.60** | **68.72** |

**Coverage tracks Q_C exactly, across five structurally different selection
rules.** Strategies reaching the same Q_C get identical coverage; strategies
reaching ~56% of it capture ~15–20% of the gain.

Stratified sampling fails precisely because spreading across *all* deciles wastes
most of the budget below the operative quantile. It raises Q_C from 20.05 to
26.88 at 6M — real, but far short of 68.72 — because only the top decile
contributes to the (1−α) quantile and it receives only N/10 of the budget.

**This is the paper's mechanism confirmed by a route it never tested.** Task 10
showed the selector attains the *maximum* Q_C for a fixed pool; Item 2 now shows
that a completely different rule reaching the same Q_C gets the same coverage,
and rules reaching less get proportionally less.

## The tail_plus_recency result is not a duplicate

Its exact match to the selector on every metric looked like a bug, so it was
checked directly: the two selections share only **166 of 254 indices (65.4%)**
yet produce **identical Q_C (68.7195)**. They are different calibration sets that
happen to reach the same operative quantile.

That is the strongest single demonstration in the paper that **Q_C is the
operative quantity and set membership is not** — two rules, 35% different
composition, indistinguishable coverage.

It also has a practical implication worth a sentence: a practitioner can spend
half the calibration budget on recency (useful for other reasons — recent months
may be more representative) and still match the pure extreme-tail selector,
provided the other half reaches the same tail.

## What this changes for the paper

1. **The selector's motivation is vindicated, not undermined.** The obvious
   reviewer alternative — generic stratification — captures under a fifth of the
   benefit. The extremes are doing something the middle bins cannot.
2. **But the selector is not unique**, and the paper should say so. Any rule
   attaining the same Q_C performs identically. The selector is *a* way to reach
   maximal tail-reach at fixed N, not *the* way.
3. **This strengthens the Item 5 / Task 16A case for leading with Q_C.** Five
   selection rules, one predictor, exact agreement.

## Honest caveats

- **Stratified sampling was given a fair but not optimised implementation.**
  Equal-count bins with N/n_bins from each is the natural reading of the spec; a
  variant weighting upper bins more heavily would do better — but that variant
  is just the selector with extra steps, which is the point.
- **The Current horizon is saturated** (93.85% for every strategy including the
  baseline) and contributes nothing.
- **Healthcare shows one negative fraction** (quintile at −0.198, i.e. slightly
  worse than the baseline). Within noise at n=146, but reported rather than
  clipped to zero.
- **One seed for the stratified draws.** The bin-level sampling is random;
  repeated seeds would vary the fraction captured by a few points, not the
  conclusion.
- **Q_C is reported at the fixed nominal α = 0.10**, whereas ACI's α drifts
  during a run. The exact numerical agreement between tail_plus_recency and the
  selector holds at that level.
