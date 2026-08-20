# Task 23 — Ensemble Disagreement as the Missing Feature

## The three answers, plainly

**(a) Did 2022 get fixed?** The question was based on a false premise. The old β
already ranked 2022 as its **highest-difficulty year (mean rank 0.993)**. Task 22's
"zero flagged cases" came from a category defined as *low-rank-with-high-error* —
2022 could not enter it by construction. Adding disagreement moved 2022's
tercile share 75%→83%, but its rank remains indistinguishable from 2023/2024
(0.970–0.978), which had 5–8× lower error. So: nothing needed fixing there, and
nothing meaningfully was.

**(b) Is there a real aggregate result?** No. The augmented method produces the
best numbers of any arm across Tasks 21–23 — narrowest width (51.25 vs 53.29),
better Winkler mean *and* median than DtACI, no vacuity — but win rate is 45.8%
and BH q = 0.468. Not validated.

**(c) Are these the same answer or different?** **Different, and both negative
for different reasons.** Item 3 is negative because the diagnosed gap was
misdiagnosed. Item 4 is negative because a directionally-correct effect is too
small to establish at n=59. Per the guardrail, neither is allowed to stand in for
the other.

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — Legitimacy | **All 3 PASS** | Row-local (unchanged under past *and* future corruption), fully continuous (635/635), non-redundant (worst R²=0.887 vs 0.95 bar). |
| 2 — Re-validation | **1/4, still 6M** | Held-out r improved at **every** horizon; perturbation percentile **collapsed** at Current/1M. |
| 3 — 2022 direct | **Premise wrong** | Old β already ranked 2022 highest (0.993). "Blind spot" was a definitional artifact. |
| 4 — Comparison | **Not validated** | Best-looking arm yet, but 45.8% win rate, BH q=0.468. |

## The most interesting finding: the two checks disagreed, and that was informative

Adding disagreement improved held-out correlation at all four horizons — Current
nearly tripled (0.109→0.302), 6M reached 0.539. On correlation alone, three new
horizons would have looked newly viable.

But the perturbation percentile **collapsed** at Current (0.370→0.025) and 1M
(0.595→0.030) — near the bottom of the random-direction null. The reason is
mechanical: at those horizons β put 18.7% and 25.6% of its weight on
disagreement, which is an *output* of the base models, not a direction the
ensemble's prediction is sensitive to. β became a better **correlational**
predictor and a worse **causal-adjacent** direction at the same time.

Task 21's Check 2 exists for exactly this ("correlational fit alone is not
sufficient evidence β captures something causal-adjacent rather than a fitting
artifact"). Inheriting Task 21's gate instead of re-running it, as this task
required, would have hidden this entirely.

6M is the exception: only 11.5% disagreement weight, r_cal 0.539, and a
perturbation percentile of a perfect **1.000** — β beat all 200 random
directions. It is the one horizon where the feature adds content without
hijacking the direction, and Item 1 predicted this (6M had the most independent
disagreement content, R²=0.675 vs 0.84–0.89 elsewhere).

## Task 22's trap did not recur

Worth stating explicitly given the task's framing:

| | Task 22 | **Task 23** |
|---|---|---|
| width/spread | 134.6 **(vacuous)** | **86.0** (below both baselines) |
| Winkler median vs DtACI | +33.2 worse | **−0.39 better** |
| win rate | 28.8% | 45.8% |
| BH-significant | Yes (q=0.0033) | No (q=0.468) |

Task 22 gamed the 20× miss penalty on four months at the cost of the other 55.
Task 23 does the opposite: genuinely narrower intervals, better typical month, no
inflation — and no significance. This is a better failure than Task 22's
apparent success.

## Discipline notes

- **A self-referential leak-grep produced a false FAIL** in Item 1 (the only
  matching line was the check's own source). Third occurrence of this bug class
  across Tasks 20/22/23; the substantive corruption tests passed throughout.
- **Item 3's mechanical answer was YES; the honest answer is no.** Reporting the
  75%→83% tercile gain without checking whether 2022's rank *discriminates* from
  low-error years would have been a clean-looking false positive.
- **Wilcoxon was deliberately omitted** from Item 4. It reported p=0.0022 on
  these data in Task 22 while permutation tests said 0.34–0.82.

## Scope

6M only, recession testbed only. Task 21's gate was re-run fresh, not inherited,
and returned the same 1/4 verdict by a different route.

## Files

- `task23_1_disagreement_check.py` / `.md`, `task23_1_disagreement.csv`, `task23_1_leakcheck.txt`
- `task23_2_validation.py` / `.md`, `task23_2_validation.csv`
- `task23_3_2022_direct.py` / `.md`, `task23_3_2022_direct.csv`
- `task23_4_comparison.py` / `.md`, `task23_4_comparison.csv`, `task23_4_significance.csv`
