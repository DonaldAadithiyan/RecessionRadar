# Task 13 — Temporal Generalization of the Calibration-Diversity Finding

**Status: specification. Design decisions are settled here before code, because
several are contestable and one of them determines whether the result means
anything at all.**

## Why this task exists

Every recession result in the paper rests on **one fixed split**: train on
pre-2020, test on post-2020. The finding has been validated across domains
(Task 1), model families (Task 8), baselines (Task 7), and out-of-fold scoring
(Task 3) — but never across **time within the same domain**.

The reviewer objection this pre-empts is sharp and obvious once thought of:

> *Did the calibration-diversity relationship only hold because of the specific
> episode you happened to test on? The post-2020 window contains COVID — a
> once-in-a-century shock — followed by the 2022–23 tightening cycle. A
> mechanism about "rare-event scarcity in the calibration set" is exactly the
> kind of thing that could be an artifact of testing on the most extreme
> rare-event period in the series.*

This is more damaging than a missing baseline, because it questions whether the
paper's central diagnostic generalises at all rather than whether its method is
best-in-class. Task 9A partially gestured at this by extending the window
forward, but that added only calm months to the *same* episode. This task tests
different episodes.

**Goal:** re-run the core diagnostic at several historical cutoffs and report
whether support-width diversity still outpredicts rare-event count at each. The
task is not scoped to produce a particular answer — a finding that the
relationship is COVID-specific would be a major, publishable correction and must
be reported as such.

---

## Design decisions (settled here, not left to the implementer)

### D1 — Which cutoffs, and why

Use cutoffs that place a **genuine recession** in the test window, since the
diagnostic is about rare-event scarcity and a test window with no rare events
cannot discriminate (established in Task 9A: 14 calm months moved nothing).

US recessions in the sample period (NBER): 1969–70, 1973–75, 1980, 1981–82,
1990–91, 2001, 2007–09, 2020.

Proposed cutoffs, each giving a train period ending before a recession and a
test window containing one:

**Feasibility was checked against the data before fixing this list** (65-month
test windows, rare = smoothed recession probability ≥ 50, the paper's threshold):

| Cutoff | Train months | Rare in train | Rare in test-65 | N_cal @40% | Usable? |
|---|---|---|---|---|---|
| 1989-01 | 263 | 29 | **5** | 105 | yes |
| 1999-01 | 383 | 34 | **0** | 153 | **NO — see below** |
| 2006-01 | 467 | 34 | **16** | 186 | yes (best powered) |
| 2020-01 | 635 | 50 | **2** | 254 | yes (published split) |

**The 1999 cutoff is dropped, and the reason is itself worth a sentence in the
paper.** The 2001 recession never exceeds **31.3%** in this smoothed-probability
series, so its test window contains *zero* months above the ≥50 threshold the
paper uses to define a rare event. A cutoff with no rare events in its test
window cannot discriminate between diversity and rare-count as coverage
predictors — Task 9A already established this empirically when 14 calm months
moved nothing.

Rare months (≥50) by decade, which shows how unevenly they fall:

| Decade | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|---|---|---|
| Rare months | 0 | 12 | 17 | 5 | 16 | 0 | 2 |

**Final cutoff list: 1989-01, 2006-01, 2020-01** — three cutoffs, two genuinely
new, spanning the 1990–91 recession, the 2007–09 financial crisis, and COVID.
The published split is retained so new results are comparable to the paper's
existing numbers rather than to a re-derivation.

**Rejected alternative:** evenly-spaced cutoffs (2015/2017/2019 as originally
suggested). Those produce test windows that are either recession-free (the 2010s
contain *zero* rare months) or overlap the same COVID episode — precisely the
degeneracy this task exists to avoid. Spacing must follow the rare events, not
the calendar.

**Honest consequence of only three cutoffs:** this is a weaker test than four
would have been, and the sample of episodes is small. Three is what the series
supports at the paper's own rare-event definition; claiming more would require
lowering the threshold, which would change what "rare" means mid-paper.

### D2 — What gets refit at each cutoff (the decision that matters most)

**Everything downstream of the cutoff must be refit.** Specifically:

- Stage 1 / Stage 2 forecasting models: **refit** on that cutoff's training
  period only.
- STL decomposition, rolling statistics, anomaly thresholds: **recomputed** on
  the training period only.
- Out-of-fold nonconformity scores: **regenerated** by rolling-origin CV inside
  that training period.
- Calibration sets and the selector: **rebuilt** from those scores.

This is a genuine retrain, and it is the one place in Tasks 7–13 where that is
correct rather than forbidden: the no-retraining guardrail existed to keep
baselines comparable on fixed scores, but here the entire point is to ask
whether the mechanism reappears when the pipeline is rebuilt from scratch at a
different point in history.

**The leakage trap to avoid:** the saved ensemble
(`fix-reg/models/full_chain_stacking.pkl`) was fit on data through 2019-12. Using
it to score a 1990 test window would leak 30 years of future information. It
must not be used for any cutoff before 2020. Use the surrogate RegressorChain
architecture from `task_oof_and_probit.py`, refit per cutoff, for all cutoffs
including 2020 — so the comparison across cutoffs is like-for-like.

**Consequence to state plainly in the write-up:** because the 2020 arm now uses
a refit surrogate rather than the published ensemble, its numbers will not match
Table 8 exactly. That is the price of internal comparability, and it is the same
tradeoff Task 9A documented for its rebuilt panel.

### D3 — What is actually measured

At each cutoff, reproduce the Phase 1 diagnostic — **not** the full baseline
horse race, which would be 4× the compute for a question this task is not
asking:

1. 200 random fixed-size calibration draws from that cutoff's training pool.
2. Per draw: support width (p95−p5), rare-event count, ACI coverage.
3. Report ρ(support, coverage) vs ρ(rare-count, coverage), R² for each, and the
   within-support-tertile redundancy check.
4. Report the selector's coverage vs the trailing baseline at that cutoff, so
   the *method*'s temporal stability is visible alongside the diagnostic's.

The headline number is **ρ(support) − ρ(rare)** at each cutoff. If that gap
stays positive across all four, the mechanism is temporally general. If it
collapses or inverts at any cutoff, that is the finding.

### D4 — Calibration size N at each cutoff

Training pools differ in length across cutoffs (1989 has ~263 months, 2020 has
635). A fixed N=254 would be nearly the entire pool at the earliest cutoff,
leaving no room for the 200 draws to vary — which would artificially flatten
the diversity relationship.

**Use N = 40% of that cutoff's training pool**, matching the proportion the
published split uses (254/635 ≈ 40%). Report the absolute N per cutoff so the
reader can see it varies and why.

**Verified before adopting.** The risk with a proportional rule is that at the
smallest pool, N becomes so large a share that every draw looks alike and the
diversity relationship flattens artificially. Expected rare-event count per
draw, with its hypergeometric spread:

| Cutoff | Pool | N (40%) | Rare per draw (mean ± 2sd) |
|---|---|---|---|
| 1989 | 263 | 105 | 11.6 (6.6 – 16.6) |
| 2006 | 467 | 186 | 13.5 (8.0 – 19.0) |
| 2020 | 635 | 254 | 20.0 (13.3 – 26.7) |

All three retain substantial draw-to-draw variation in rare count, which is the
variation the diagnostic correlates against coverage. The rule is safe.

---

## Protocol

1. Rebuild the feature panel once from raw FRED series (reuse
   `task9a_extend_window.py`'s construction), then for each cutoff recompute
   STL/rolling/anomaly features **using only that cutoff's training rows**.
2. Refit the surrogate chain per cutoff; generate out-of-fold scores by
   rolling-origin CV within the training period.
3. Run the 200-draw diagnostic sweep (reuse `domain_common.random_draw_sweep`
   and `sweep_diagnostics` — do not reimplement).
4. Run the selector vs trailing-baseline comparison at each cutoff.
5. Wilson intervals on every coverage figure, per the standing guardrail.

## Deliverables

- `fix-reg/task13_temporal_cutoffs.csv` — one row per (cutoff × horizon):
  `cutoff, n_train, N_cal, n_test, rare_in_test, rho_supp, R2_supp, rho_rare,
  R2_rare, within_tertile_rho_rare, cov_mean, cov_std`.
- `fix-reg/task13_selector_by_cutoff.csv` — selector vs trailing baseline
  coverage and width at each cutoff, with Wilson intervals.
- `fix-reg/task13_temporal_generalization.md` — write-up in the established
  format, containing:
  - a per-cutoff table of ρ(support) vs ρ(rare) with the gap highlighted;
  - an explicit verdict: does the diagnostic hold at every cutoff, some, or only
    the published one;
  - the regime character of each test window (which recession, how severe, how
    many rare months) so a reader can see what each cutoff actually tested;
  - honest caveats, including the D2 comparability consequence.

## Guardrails

- **The 2020 arm must use the refit surrogate, not the saved ensemble.**
  Comparing a leak-free 1990 result against a leaky 2020 result would invalidate
  the whole exercise.
- Earlier cutoffs have shorter training pools and fewer rare events; report
  those counts alongside every correlation, since a weak ρ at a cutoff with 3
  rare months means something different than a weak ρ at one with 18.
- Do not report a coverage number without its Wilson interval.
- If the relationship weakens at earlier cutoffs, do not attribute that to data
  quality without checking it — report the correlation, the pool size, and the
  rare-event count, and let the reader weigh them.
- If any cutoff cannot be run (insufficient training history, missing series
  coverage in the early period), report it as not-run with the reason, per the
  convention used for CPTC in Task 7 and the HOSPITAL components in Task 8.

## Suggested order of work

1. **2020 cutoff first**, with the refit surrogate — the only one whose answer is
   roughly known, so it validates the harness before the novel cutoffs run. If
   the refit surrogate does not roughly reproduce the published diagnostic
   (ρ(supp) ≫ ρ(rare) at 6M), that is a harness bug to fix before trusting
   anything else — the same discipline as Task 7's Phase-3 reproduction gate.
2. **2006 next** — 16 rare months in test, the best-powered new cutoff and the
   most informative single result in the task.
3. **1989 last** — shortest pool (263 months) and only 5 rare months in test, so
   the most likely to produce an ambiguous or null answer for power reasons
   rather than substantive ones. Interpret accordingly.

## What would count as each outcome

Stated in advance so the result cannot be reinterpreted after the fact:

- **Mechanism is temporally general:** ρ(supp) > ρ(rare) at all three cutoffs,
  with the gap comparable in sign and rough magnitude to the published split.
- **Mechanism is episode-specific:** the gap collapses or inverts at 1989 and/or
  2006 while holding at 2020. This would be a major correction and would require
  reframing the paper's central claim as conditional on the shift's character.
- **Underpowered / inconclusive:** gaps are directionally consistent but with
  wide scatter at the smaller cutoffs. Report as such; do not round toward
  either conclusion.

Data coverage was verified: all 12 base series are non-null from 1967-02, so no
cutoff fails for missing early data.
