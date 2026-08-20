# Task 22 — Fixing the Three Diagnosed Bugs in the Error-Direction Method

## Does fixing these three bugs produce a defensible result?

**No. Outcome 3, same as Task 21 — but for a different and more informative
reason than last time.** Both fixes were implemented correctly and both did what
they were designed to do. Neither produced an improvement that survives scrutiny.
The method still does not beat DtACI in any defensible sense at 6M on this
testbed, and the scope was never larger than that.

The one genuinely useful thing this task produced is a clean decomposition of
*why*: **the confound fix was correct and inert; the ceiling fix bought a
statistically-significant mean improvement with a 50.5% width increase that
pushes the intervals past this project's own vacuity threshold.**

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — Rolling recentering | **Fixed, leak-free, ineffective** | Saturation 15.4% → 3.1%; mean multiplier 1.156 → 1.143 against a 1.000 target. Statistically inert (all p ≥ 0.64). |
| 2 — Raised ceiling | **Fixed from fit data only** | HI = p99/p90 = **2.79** (was 1.25). Independently clears the 1.76 the worst 2020 miss needed. |
| 3 — 2022 blind spot | **Diagnosed, no fix** | **CASE 2** — within the fit distribution, a catchable miss. Not extrapolation failure. |
| 4 — Re-run comparison | **Run, outcome 3** | Mean Winkler beats DtACI (−2.57, BH q=0.0033) but **win rate 28.8%**, width +50.5%, Winkler median +33.2 worse. |

## The result that matters most

`task22_both` produces a *statistically significant* mean-Winkler improvement
over DtACI. It is still not a real improvement:

- **Win rate 28.8%** — it loses in 71.2% of individual months.
- **Winkler median worsens by 33.2** while the mean improves by 2.57.
- **Width +50.5%**, giving a width/spread ratio of **134.6**, past the ~100
  vacuity threshold this project already uses. It bought 96.61% coverage with
  intervals wider than the target's plausible range.
- **Both permutation nulls are non-significant** (0.821 and 0.342 against DtACI)
  despite Wilcoxon p=0.0022. Wilcoxon assumes exchangeable pairs; these are a
  time series with four enormous 2020 outliers. The autocorrelation-aware tests
  do not confirm it.

The mechanism is transparent: Winkler penalises a miss at 20× the shortfall, so
widening the four Jan–Apr 2020 intervals avoids four huge penalties and rescues
the mean, while the other 55 months pay 50% extra width for nothing.

This is the same trap Task 21 caught at a 40.7% win rate — but here it arrives
*with* a significant p-value and a BH-corrected q of 0.0033. That makes it more
dangerous, not less. Reporting the mean and the BH q alone would have made this
task look like a success.

## What Item 3 found, and why it closes off the obvious next move

2022 is **CASE 2**: its inputs sit squarely within the fit distribution (fewer
beyond-3SD cells than average, in-line centroid distance, identical
projection-in-range), yet the base model was wrong on them. β maps inputs →
expected error, so when the base model fails on ordinary-looking inputs there is
nothing in *x* to signal it. **No ceiling and no recentering can fix this** — the
information isn't in the feature vector.

That matters for scoping: the natural next instinct after Task 22 would be
another tweak to the response function. Item 3 says that will not help. A real
fix needs a signal source β currently ignores — ensemble disagreement is the most
promising, since it is already computed here (Task 4) and is outcome-free at
prediction time. Per the guardrail, that is a recommendation for a future scoped
task, not a patch applied here.

## Discipline notes

Three places where the convenient answer was available and rejected:

- **A tuning opportunity declined.** A bounded reference window (60 months) cuts
  the multiplier deviation from 0.143 to 0.094 — the best of five options. Picking
  it because it minimises deviation *on the test period* is test-set tuning. The
  expanding window has no free parameter and is what ships; the residual is
  reported.
- **The ceiling was derived, not chosen.** HI = p99/p90 from the 291-month fit
  split, with no reference to the 2020 cluster. It landing above the 1.76 that
  cluster needed is reported as an observation, not validation.
- **Item 3's first verdict was wrong in the convenient direction.** An absolute
  criterion flagged CASE 1 (extrapolation — which would have excused the blind
  spot). Every test year failed that criterion, so it discriminated nothing.
  Corrected to a relative comparison, the verdict reverses to CASE 2.

Two analysis bugs found and fixed along the way: a naive leak-grep that matched
legitimate training-data usage and its own source line, and two constant-in-fit
features that produced 1e12 z-scores and made the first 2022 comparison
meaningless.

## Scope

Everything here is **6M only, recession testbed only**. Task 21's validation gate
was not touched — Current, 1M and 3M remain unvalidated and were never run.

## Files

- `task22_1_recentering.py` / `.md`, `task22_1_recentering.csv`, `task22_1_leakcheck.txt`
- `task22_2_ceiling.py` / `.md`, `task22_2_ceiling.csv`
- `task22_3_2022.py`, `task22_3_2022_diagnosis.md`, `task22_3_2022.csv`
- `task22_4_comparison.py` / `.md`, `task22_4_comparison.csv`, `task22_4_significance.csv`
