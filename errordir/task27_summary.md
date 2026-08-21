# Task 27 — Tail-Targeted β, and a Missing Baseline

## The pre-registered criterion was NOT met, on either domain. And no — β_tail does not beat every existing baseline (including CQR) on both coverage and width simultaneously, on either domain. Neither does β_mean.

The direct question first, since the task asks for it plainly: on **climate**,
errordir_mean leads on Winkler but diversity_optimal beats it on coverage (94.65
vs 91.36) and bellman_ci on width (7.552 vs 9.441). On **energy**, errordir_tail
leads on Winkler but diversity_optimal again beats it on coverage (89.50 vs
86.25) and bellman_ci on width (3.385 vs 7.628). **No arm dominates both axes on
either domain.** Adding CQR did not change this — CQR places 9th of 11 on
climate and 5th of 11 on energy.

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — CQR baseline | **Added, correct** | Exactly 90.0% on a synthetic control. 9th of 11 (climate), 5th of 11 (energy). |
| 2 — β_tail + gate | **1 of 2 validates** | Passes on energy; **fails on climate** at percentile 0.030 — a **40× collapse** in perturbation effect (0.0155 vs β_mean's 0.625). |
| 3 — Head-to-head | **Criterion not met** | Energy: coverage gap **identical** to 3 s.f. (3.25 pp). Climate: gap *widened* and Winkler advantage halved. |
| 4 — Direct verdict | **No universal dominance** | Stated above. |

## Why tail-targeting failed to move the coverage gap

The multiplier is a **monotone rank map** of the projection, so coverage depends
on the *ordering* of test points by projected difficulty, not the projection's
scale. β_mean and β_tail induce nearly the same ordering — features predicting
large mean error largely coincide with features predicting tail membership. On
energy the coverage gap came out at **3.25 pp under both**, to three significant
figures.

This locates the binding constraint elsewhere: the **±25% multiplier band**, not
the fitting target. Closing a 3.25 pp gap needs high-difficulty intervals widened
beyond what the band allows — and Task 22 already tested raising that ceiling,
which produced vacuous intervals (width ratio 134.6) and gamed the Winkler mean
via four outlier months. **The frontier position looks structural, not a fixable
design choice.** That is the most useful thing this task establishes.

## Two findings worth keeping beyond the headline

**1. The decision to reuse the existing gate was vindicated.** The task rejected
a new tail-relevance gate as circular. β_tail's climate failure shows why: its
held-out correlation with realized error is fine (0.281 vs β_mean's 0.333), but
perturbing along it barely moves the prediction (40× collapse). A gate asking
"does moving along β_tail change P(tail|x)" would very likely have **passed** —
β_tail was fit to predict exactly that — and would have concealed the collapse in
the quantity that actually builds intervals. This is the third instance of the
correlational-vs-causal-adjacent gap in this project (Task 23's disagreement
feature, Task 25's tree models, now β_tail).

**2. CQR's energy result is informative rather than embarrassing.** CQR
under-covers badly there (**56.00%** against a 90% target). I verified this is
real, not an implementation error: the energy test period is a different price
regime (test mean 7.39 vs FIT 6.40, test p90 11.68 vs FIT 9.14). Split-CQR's
guarantee is conditional on **exchangeability**, which fails under distribution
shift, and it has no adaptive mechanism to recover — unlike the ACI-family
methods, whose α updates online. A gradient-boosted quantile variant does better
but still under-covers (75.75%). This is a concrete demonstration of why the
adaptive methods this project studies exist.

## Standing claim, unchanged

> The error-direction method achieves the best combined coverage-width tradeoff
> (Winkler) among eleven methods on climate (n=243) and energy (n=400),
> significantly and without gaming. It does not dominate any single axis, does
> not validate for tree-based models, and — new from this task — **fitting β
> toward tail probability rather than mean error does not change this.**

## Scope note

CQR was run on climate and energy, the two domains carrying a real non-null
errordir result and the two where Item 3's head-to-head is defined. Adding it
retroactively to the recession/healthcare/insurance tables would require
re-running those pipelines; recorded as a scope limitation, not claimed as
complete.

## Files

- `task27_1_cqr.py` / `.md` — CQR implementation, synthetic control, results
- `task27_2_beta_tail.md`, `task27_2_validation.csv` — tail fit and gate
- `task27_3_comparison.py` / `.md`, `task27_3_comparison.csv`,
  `task27_3_significance.csv`
