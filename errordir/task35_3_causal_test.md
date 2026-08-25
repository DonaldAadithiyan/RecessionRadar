# Task 35, Item 3 — Causal-Adjacency Test on the Locality-Confirmed D(x)

## Result: **Check 2 fails on all six fits (0/6), including both ridge controls. HOS on ridge splits: energy passes (percentile 1.000), insurance fails (0.800).**

Script: `task35_3_causal_test.py` · Data: `task35_3_causal_test.csv`

All six fits passed Item 2's locality gate, so all six were tested. Bandwidth is
Silverman, as fixed in Item 1 and used in Item 2.

## Results

| domain | model | class | Check 1 | **Check 2 pct** | C2 pass | **HOS** | HOS pct | HOS pass |
|---|---|---|---|---|---|---|---|---|
| insurance | **ridge** | linear | Pass | 0.504 | **No** | 0.047 | 0.800 | **No** |
| insurance | xgboost | tree | Pass | 0.461 | **No** | — | — | — |
| insurance | lightgbm | tree | Pass | 0.485 | **No** | — | — | — |
| energy | **ridge** | linear | Pass | 0.540 | **No** | **0.418** | **1.000** | **Yes** |
| energy | xgboost | tree | Pass | 0.378 | **No** | — | — | — |
| energy | lightgbm | tree | Pass | 0.406 | **No** | — | — | — |

Instrument routing as fixed by the task: Check 2 on every fit; HOS additionally
on ridge only (Task 37 verified it valid on linear models); **DS not used
anywhere** (Task 36: degenerate on linear models, no working tree correction).

## Check 2's tree-model limitation, stated with the result

Per the task's requirement: Check 2 on tree fits carries the **unresolved
hyper-responsive-null limitation** documented in Task 33 — trees respond to
essentially every perturbation direction (zero-response rate 0.00003, 2.7× ridge's
median change), so the random-direction null is high and tight, and a candidate
direction has little room to separate by magnitude percentile. Task 34 attempted
a correction (DS) which was verified on trees but then found degenerate on linear
models, and no tree-specific replacement was completed.

**So the four tree Check 2 failures here are not fully conclusive on their own.**
They are consistent with every prior tree result (0/20 across Tasks 32–34) but
inherit the same instrument limitation.

## The ridge controls are the load-bearing result

Check 2 fails on **both ridge fits** — 0.504 and 0.540, essentially chance. This
matters more than the tree results, because Check 2 on ridge is a *validated*
instrument: Task 37 Item 1 confirmed it is not trivially satisfied (fitted-but-
meaningless directions score 0.18–0.48 and pass 0/20, while β scores 0.995).

So a ridge Check 2 failure is informative, and it says the Silverman-bandwidth
∇D(x) is **not** causally adjacent on linear models either. For direct
comparison, β on the same ridge fits scores **0.995** under the identical test.

## The HOS split, and what it does and does not show

HOS was run on D(x) **as a ranking score** (not its gradient), since that is the
difficulty-ranking claim D(x) makes.

- **energy/ridge: HOS 0.418, percentile 1.000 — passes.** D(x) orders held-out
  points by realized error better than all 200 permutation-fitted D's. For
  reference, β on energy/ridge scored HOS 0.402 (Task 37), so D(x) is
  *marginally better than β* at ranking.
- **insurance/ridge: HOS 0.047, percentile 0.800 — fails.** Essentially no
  held-out ranking signal.

This is a genuine split, and the honest reading is that D(x) carries real ranking
information on energy and not on insurance. It is **not** evidence of causal
adjacency — HOS is explicitly correlational (Task 37), and the thing Item 3 was
built to test is Check 2, which failed everywhere.

## Summary of what this item established

- The locality fix from Items 1–2 was real: gradients now genuinely vary.
- A genuinely-local ∇D(x) still **fails causal adjacency on every fit**,
  including the validated ridge control.
- D(x)-as-a-ranking-score retains useful signal on one domain of two.
