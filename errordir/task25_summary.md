# Task 25 — Practical Models, New Domains: Does This Generalize Beyond Ridge?

## The two questions, answered separately

**Beyond ridge? NO.** All four tree fits (XGBoost, LightGBM × 2 datasets) fail
Check 2. Combined with Task 24, the split across ten fits is perfect by model
class: **linear 4/4 pass, tree 0/6 pass.**

**Beyond climate? PARTIALLY YES.** Energy/ridge wins the combined Winkler
tradeoff (1st of 9, n=400), replicating climate/ridge's result on a genuinely
new domain. Insurance/ridge does not — it places 2nd behind Mondrian.

These are different answers and neither substitutes for the other.

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — Fix Check 2 for trees | **Prescribed fix does NOT work** | The premise was wrong: the tree null is *lower*-variance, not spiky. Same-class and magnitude-matched nulls both still fail. |
| 2 — Climate/gradboost retest | **Still fails (0.035)** | Task 24's open item resolved as a negative, not a rescued positive. |
| 3 — Dataset selection | **2 selected, 2 rejected** | Insurance severity (26,444 claims, p99/p50=14.3) and energy prices (45,264 half-hours, 0.37% spikes). |
| 4 — Baseline suite | **Full 8 baselines, both datasets** | Uniform calibration split for every method. |
| 5 — Validation | **2/6 validated, both ridge** | Ridge passes at pct 1.000 on both; all trees fail at 0.32–0.52. |
| 6 — Comparison | **Energy 1st of 9; insurance 2nd** | Neither beats the best baseline on coverage and width simultaneously. |

## Item 1: the prescribed fix does not work, and the premise was incorrect

The task asserted trees fail because "step functions move more under random
perturbation, inflating the null's p95." Measured directly, the gradboost null
has **lower** variance and **less** skew than ridge's, with **zero** zero-effect
directions:

| model | mean | p95 | sd | frac zero |
|---|---|---|---|---|
| ridge | 0.227 | 0.553 | 0.167 | 0.000 |
| gradboost | 0.908 | 1.103 | **0.119** | **0.000** |

A second candidate confound (β sparse vs random directions dense) was tested and
rejected — 9.7 vs 8.0 effective dimensions.

Three null constructions were built and verified against the known
climate/gradboost case:

- **(A) same-model-class** — which **Task 24 already used**, verified rather
  than assumed. Still fails: 0.035.
- **(B) magnitude-matched** — isolates direction from step-size. Still fails: 0.030.
- **(C) sign-structure** — passes (1.000), but **rejected**. It asks whether a
  regression fit to predict |error| beats random projections at correlating with
  |error| — the objective it was optimized for. Any generalizing β passes. It
  also *breaks* healthcare/ridge (0.910), so it is not a repair but a different,
  weaker test, and adopting it after seeing which cases it rescues would be
  post-hoc selection. It would also delete the check that produced Task 23's
  most informative finding.

**Conclusion: Check 2 is not broken for trees. It reports something real** — β,
a linear projection, is a weak lever on a piecewise-constant predictor whose
influence concentrates on a few axis-aligned splits.

## The ridge control arm was the decisive design choice

Including ridge alongside the two practical models separated two hypotheses that
Task 24 could not distinguish: *tree model class* vs *these particular domains
lack signal*. Same data, same features, same procedure — ridge passes at 1.000
on both datasets while XGBoost and LightGBM sit mid-null.

The sharpest case: **energy/LightGBM has the highest held-out correlation of any
fit in this task (0.437, above ridge's 0.417) and still fails at 0.520.** Strong
correlation, weak causal adjacency — the Task 23 pattern, now replicated on
fresh data at n=400.

## What generalizes and what doesn't

Energy/ridge is a genuine replication of climate/ridge: 1st of 9 on Winkler,
beats six of eight baselines on coverage **and** width simultaneously,
non-vacuous (ratio 0.61), significant against five baselines after BH.

It falls short in the same way climate did — no per-axis dominance. acmcp has
higher coverage at a 5.92 width ratio (10× wider); bellman_ci is narrower at 68%
coverage. And win rate is below 50% against dtaci (25.2%), mondrian (21.0%) and
bellman_ci (27.8%), where the mean advantage is carried by the spike tail and is
not significant against the first two.

Insurance is the weaker case: 2nd of 9, losing to Mondrian (16.5% win rate,
q=0.880).

**One pattern worth naming, not yet a result:** Mondrian has now beaten errordir
on both non-temporal randomly-split datasets (healthcare, insurance) and lost on
both temporal ones (climate, energy). n=4, but a specific testable hypothesis.

## Standing claim across Tasks 21–25

> The error-direction method achieves the best combined coverage-width tradeoff
> (Winkler) among nine methods on **two temporal domains** — climate (n=243) and
> energy (n=400) — significantly and without gaming, beating five to six
> baselines on coverage and width simultaneously. It does **not** dominate any
> single axis, does **not** validate for tree-based models (0/6 fits), and loses
> to Mondrian on both non-temporal datasets tested.

## Files

- `task25_1_treenull_fix.py` / `.md`, `task25_1_treenull_fix.csv`
- `task25_2_gradboost_retest.md`
- `task25_3_dataset_selection.md`
- `task25_build.py`, `task25_456.py`
- `task25_4_baselines.csv`, `task25_5_validation.csv` / `.md`
- `task25_6_comparison.csv` / `.md`, `task25_6_significance.csv`
