# Task 29, Item 2 — Original errordir vs RLCP, Direct

## Climate: **statistical tie.** Energy: **errordir clearly better, but RLCP breaks down there rather than losing a fair fight.**

Script: `task29_2_comparison.py` · Data: `task29_2_comparison.csv`,
`task29_2_significance.csv`

errordir configuration = the winning one: **β_mean, rank-based fixed multiplier
[0.75, 1.25], rolling recentering**. Not the q_α(z) redesign. Both methods use
the identical FIT/CAL/TEST partition — same β-fitting split, same calibration
scores, same test stream (Task 24 Finding 3).

## Climate (n=243)

| method | coverage | width | ratio | unbounded | **Winkler mean** | Winkler median |
|---|---|---|---|---|---|---|
| **errordir** | 91.36 | 9.441 | 1.08 | 0 | **11.220** | 9.528 |
| rlcp (published) | 91.77 | 9.631 | 1.10 | 0 | 11.292 | 9.642 |
| rlcp_best_gamma | 91.77 | 9.631 | 1.10 | 0 | 11.292 | 9.642 |
| *baselcp (Task 28's)* | *90.95* | *9.035* | *1.03* | *0* | *10.723* | *8.873* |

| comparison | mean diff | win rate | perm p | **BH q** |
|---|---|---|---|---|
| errordir vs **rlcp** | −0.072 | 54.7% | 0.398 | **0.556** |
| errordir vs rlcp_best_gamma | −0.072 | 54.7% | 0.417 | 0.556 |
| errordir vs baselcp | **+0.497** | 22.6% | 0.986 | 0.986 |

**errordir and RLCP are a statistical tie on climate.** A Winkler difference of
0.072 on ~11.2 (0.6%), a win rate of 54.7% barely above chance, and q=0.556.
Coverage differs by 0.41 pp, width by 2%. There is no defensible sense in which
either beats the other here.

**baseLCP actually beats both** (10.723). Worth stating plainly: the
un-randomized variant scores better on this Winkler comparison than the published
randomized one. That is not a knock on RLCP — the randomization and the `+∞` atom
buy *validity guarantees*, which cost sharpness. It does mean Task 28's reported
"RLCP tie" was a tie against a different, more aggressive estimator.

## Energy (n=400)

| method | coverage | width | ratio | **unbounded** | Winkler mean | Winkler median |
|---|---|---|---|---|---|---|
| **errordir** | **86.25** | 7.395 | 0.61 | **0** | **28.775** | 6.974 |
| rlcp (published) | 76.75 | 3.831 | 0.31 | **27** | **undefined** | 4.233 |
| baselcp (Task 28's) | 72.50 | 3.811 | 0.31 | 0 | 30.887 | 3.589 |

**RLCP produces unbounded intervals on energy at every bandwidth tested:**

| γ | coverage (raw) | unbounded | coverage among **bounded** intervals |
|---|---|---|---|
| 0.879 | 75.50 | 15 | 74.55% |
| 1.758 | 75.75 | 21 | 74.41% |
| 3.516 | 76.75 | 27 | 75.07% |
| 7.031 | 78.00 | 38 | 75.69% |
| 14.06 | 80.25 | 49 | 77.49% |

Winkler mean is undefined whenever any interval is infinite, so the headline
comparison cannot be made — that is itself the finding. Reading around it:

- RLCP's raw coverage (75–80%) **counts unbounded intervals as covered**. Among
  bounded intervals it is 74–77%, versus errordir's **86.25%** with zero
  unbounded.
- Against baseLCP, which is bounded and therefore comparable, errordir wins on
  Winkler mean (28.775 vs 30.887) — but at a **22.2% win rate** and q=0.250, so
  not significant. baseLCP wins the majority of individual hours with much
  narrower intervals and loses badly on the spikes.

The mechanism is the same one Task 27 found for CQR and Task 28 for q_α(z):
energy's test period is a different price regime, and a kernel that localizes on
calibration-period geometry finds too little effective local mass, pushing the
weighted quantile past the finite atoms into the `+∞` tail. errordir's ACI
anchor adapts online and does not have this failure mode.

## Anti-gaming discipline

Win rate reported alongside every mean; no arm vacuous (all finite ratios ≤ 1.10);
block sign-flip permutation (both domains temporal, per Task 26); BH across all
cells. The unbounded-interval cells are reported as undefined rather than filled
with a substitute statistic.
