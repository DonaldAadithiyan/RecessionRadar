# Task 28, Item 4 — Held-Out Per-Bin Coverage Check

## Verdict: **PASSES on climate with caveats, MARGINAL on energy — and it correctly predicted energy's downstream failure before Item 5 ran.**

Data: `task28_4_bins.csv`

## The non-circular design

`q_α(z)` is fit on a random half of the calibration split (**CAL-A**) and the
per-bin coverage check runs on the **disjoint** other half (**CAL-B**). A
`q_α` that merely reproduces its training data fails here. This is disjoint from
whatever fit the function, as the guardrail requires — the same circularity that
killed the tail-relevance gate in Task 27.

## Results — empirical coverage per z-quintile on held-out data (target 90%)

### Climate (fit on 275, checked on 276 disjoint)

| bin | n | z range | **coverage** | mean q_α |
|---|---|---|---|---|
| 1 | 55 | lowest | **83.64%** | 2.924 |
| 2 | 55 | | 94.55% | 3.879 |
| 3 | 55 | | 94.55% | 4.385 |
| 4 | 55 | | 96.36% | 6.177 |
| 5 | 56 | highest | 87.50% | 7.332 |

max |deviation| = **6.36 pp**, mean |deviation| = 4.86 pp.

### Energy (fit on 800, checked on 800 disjoint)

| bin | n | z range | **coverage** | mean q_α |
|---|---|---|---|---|
| 1 | 160 | lowest | **81.88%** | 0.776 |
| 2 | 160 | | 90.62% | 0.881 |
| 3 | 160 | | 90.00% | 1.272 |
| 4 | 160 | | 86.25% | 1.973 |
| 5 | 160 | highest | 89.38% | 3.085 |

max |deviation| = **8.12 pp**, mean |deviation| = 2.62 pp.

## Reading this honestly

**The function generalizes in shape.** `mean q_α` rises monotonically across bins
in both domains (climate 2.92 → 7.33; energy 0.78 → 3.09), and it does so on data
the estimator never saw. The learned "this difficulty level needs this much
width" relationship is real, not memorized.

**But calibration is imperfect, and the pattern is informative.** Both domains
under-cover in **bin 1** (83.64%, 81.88%) — the lowest-difficulty bin, where
`q_α` assigns the narrowest intervals. The isotonic fit's left-end knot is
estimated from the fewest effective observations and is clamped for
extrapolation, so the low-z end is where the estimator is weakest. Climate then
over-covers in bins 2–4 (94.6–96.4%), which is the mirror image: width shifted
away from easy cases lands on middling ones.

**This check earned its place.** Energy's bin-1 under-coverage of 81.88% is a
direct advance warning of what Item 5 found — `errordir_qalpha` collapsing to
68.50% overall coverage on energy. The validation ran first and flagged the risk
before the comparison, which is exactly the purpose of a non-circular gate. Had
this item been skipped, or run circularly on CAL-A, the energy failure would have
appeared only in the final table with no diagnosis attached.

**Verdict:** the function is usable on climate and should be treated as
**not adequately calibrated on energy at the low-z end**. Item 5 reports both.
