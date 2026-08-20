# Task 24 — Does the Error-Direction Method Work on Other Domains?

## Per-domain verdict

**Climate: validated (ridge), and produced the strongest error-direction result
in this project.** Best combined Winkler of all 9 methods (11.215), narrowest
width of any non-degenerate method (9.443), win rate above 50% against 7 of 8
baselines, BH q ≤ 0.0027, non-vacuous (ratio 1.08). **But it does not beat the
best baseline on each axis separately** — diversity_optimal has higher coverage
(94.65% vs 91.36%) and bellman_ci narrower width (7.552 vs 9.443). By this
task's own stated bar that is a **partial result**: best coverage-width tradeoff,
not frontier dominance.

**Healthcare: validated (ridge), weak result, loses to Mondrian.** 2nd of 9 on
Winkler (29.852 vs Mondrian's 25.961, win rate 30.8%). Does not beat DtACI
(47.9% win rate, q=0.291). Only clearly beats acmcp and bellman_ci, this
project's known weak baselines.

**Healthcare's outcome matches its pre-registered expectation.** It was flagged
in advance as low-headroom (baseline coverage 88–91%, "most headroom spent on
over-coverage"). Every method lands in 88–92% coverage with width ratios
0.88–0.94 — there is almost nothing to separate them, and a weak,
mostly-non-significant effect is what the domain's known properties predicted.
Stated as anticipated, not invoked to explain away the result — and not used to
soften the one clear negative, that Mondrian genuinely beats errordir there.

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — Feature audit | **Done** | Both domains have a **legitimate** disagreement analog (ridge/gb trained on identical task, corr 0.62–0.72). Structurally weaker than recession's: 1 feature not 3, not an internal ensemble component. |
| 2 — Four fits | **Done** | Climate generalizes far better (held-out r 0.335–0.374) than healthcare (0.128–0.149). |
| 3 — Fresh gate | **2/4 validated** | Both ridge fits pass; **both gradboost fits fail** Check 2 for a mechanical reason. |
| 4 — Full comparison | **Partial win (climate), loss (healthcare)** | Climate best-on-Winkler, significant, ungamed. Healthcare 2nd, loses to Mondrian. |
| 5 — Pooling | **Evaluated, recommend NO** | Climate is already powered (q≤0.0027); recession/6M was null. Pooling a positive with a null dilutes rather than strengthens. |

## Two findings that matter beyond this task

**1. Check 2 cannot currently discriminate for tree models.** Both gradboost fits
failed, but β's absolute perturbation effect was *identical* to ridge's (climate:
0.857 vs 0.850). What differed was the null — gradboost's p95 was 3× ridge's
(1.604 vs 0.530), because a step-function model moves more under random
perturbation than a linear one. The gate was applied as specified and the failing
fits did not proceed, but the honest reading is that Check 2 was **not able to
test** gradboost, not that gradboost's β failed. Climate/gradboost had the
highest held-out correlation of all four fits (0.374) and remains untested.

**2. The circular-shift permutation null is vacuous for a mean statistic.**
Rolling a vector permutes its values, so `mean(roll(d,k)) == mean(d)` exactly —
the null is degenerate. Tasks 21–23 all reported this p-value. **No conclusion
changes**: those tasks reported the block sign-flip null alongside and rested
their verdicts on it (sign-flip was non-significant throughout, matching the
reported "not validated" outcomes). Task 20's `np.roll` usage is a different,
valid construction. The shift column should be disregarded in Tasks 21–24; the
sign-flip null is the operative test.

## Where this leaves the method

Across Tasks 21–24 the error-direction method has gone from "no supportable
improvement" (recession, n=59) to a **statistically clear best-tradeoff result on
climate (n=243)** that survives every anti-gaming check Tasks 22–23 established.
That is a real advance, and it is the first result in this line of work that
does.

It is not yet "beats current methods" in the sense this project is being built
around. The honest scope is: **on climate, the error-direction method achieves
the best coverage-width tradeoff of any of 9 methods tested, significantly and
without gaming; it does not dominate any single axis, and it does not replicate
on healthcare (as predicted) or on recession.**

## Files

- `task24_1_feature_audit.md`
- `task24_2_beta_fits.py` / `.md`, `task24_3_validation.csv`
- `task24_4_comparison.py` / `.md`, `task24_4_comparison.csv`, `task24_4_significance.csv`
- `task24_5_pooling.md`
