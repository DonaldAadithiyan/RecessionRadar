# Task 35 — Locally-Varying Geometry, Properly Scoped

## **Outcome 2 — locality confirmed, causal adjacency still fails.** The bandwidth fix worked (mean gradient |cosine| 0.83–0.90 → 0.29, frac >30° from mean 4–9% → ~100%), the ridge control passed the locality gate, and a genuinely-local ∇D(x) then failed Check 2 on **all six fits (0/6), including both ridge controls**.

## What the two attempts together establish

| | Task 33 | **Task 35** |
|---|---|---|
| bandwidth rule | median pairwise distance | **dimension-adjusted Silverman** |
| h | 4.21–4.36 | **0.567–0.588** |
| effective neighbours | ~1042–1221 of 1500 | **2.2–4.3 of 1500** |
| mean gradient \|cosine\| | 0.83–0.90 (near-constant) | **0.29** |
| frac gradients >30° from mean | 4–9% | **~100%** |
| locality gate | **failed** (ridge too) | **passed** (all six) |
| Check 2 | 0/6 | **0/6** |

Task 33's negative was uninterpretable — locality was never achieved, so the
experiment never tested its own hypothesis. **Task 35 fixed that and got the same
answer.** That is the difference between an inconclusive result and outcome 2.

## Why the ridge control carries the weight

Check 2 failed on both ridge fits at **0.504 and 0.540** — chance. Check 2 on
ridge is a *validated* instrument: Task 37 Item 1 confirmed fitted-but-meaningless
directions score 0.18–0.48 and pass 0/20, while β scores **0.995** on the same
fits under the identical test.

So this is not the tree-instrument ambiguity that has clouded Tasks 32–34. On a
model class where Check 2 demonstrably discriminates, a genuinely-local ∇D(x)
scores at chance while β scores 0.995. **Locality is not the missing ingredient.**

The four tree failures are consistent with this but are reported with Check 2's
unresolved hyper-responsive-null limitation (Task 33) stated, per the task's
requirement — they are not independently conclusive.

## One positive finding, and one caveat I have to flag

**Positive:** HOS on D(x)-as-a-ranking-score splits — energy/ridge passes at
percentile **1.000** with HOS 0.418 (marginally above β's own 0.402 from Task 37);
insurance/ridge fails at 0.800 with HOS 0.047. So the local difficulty *function*
carries real held-out ranking signal on one domain of two, even though its
*gradient* is not causally adjacent anywhere. Those are different claims and the
distinction is worth preserving.

**Caveat, recorded in Item 2 before Item 3 ran:** Silverman gives only **2.2–4.3
effective neighbours** out of 1500. Measured against a random-direction baseline
at the actual dimensions, the gradient field sits just **0.3 sd (insurance)** and
**0.9 sd (energy)** above pure noise. So the field is genuinely non-random, but
weakly so. A negative Check 2 result cannot fully separate "locality doesn't
help" from "this local estimate is too noisy to help."

I did **not** re-tune the bandwidth in response. Silverman was fixed in Item 1
with its over-smoothing risk stated in advance; discovering an under-smoothing
risk afterwards and adjusting would be the test-set adaptation this project has
refused since Task 22.

## Verdict, per the guardrail

**This was the second and, absent a specific new reason, final attempt at this
mechanism.** Two bandwidth regimes have now been tested — far too wide (Task 33)
and near nearest-neighbour (Task 35). One produced no locality; the other
produced locality and no causal adjacency. Between them they bracket the useful
range, and the ridge control failing in the second case is the strongest evidence
available that the mechanism, not the tuning, is the limit.

A third attempt would need a *specific new reason* — for example, an intermediate
bandwidth chosen by a fit-period criterion that targets effective-neighbour count
directly (say 30–100 neighbours) rather than density-estimation optimality. That
is a coherent idea, but it is a new rationale rather than a retry, and this task
does not leave the door open by default.

## What is unchanged

Nothing here touches any measured Winkler, coverage, or width number, or the
standing evidence for β on ridge (Check 2 0.995 both domains, Task 37 HOS 1.000
and 0.990). This task tested an alternative mechanism and found it does not work;
it does not revise what is known about the mechanism that does.

## Files

- `task35_1_bandwidth.md` — rule and rationale, fixed before fitting
- `task35_2_locality_check.py` / `.md`, `task35_2_locality_check.csv`
- `task35_3_causal_test.py` / `.md`, `task35_3_causal_test.csv`
