# Multidomain Revision — Findings and Writing Guidance

Work completed against `multidomain_revision_spec.md`. Tasks 1–4 are done and
produced results; this document records what they found and what the Task 5/6
writing pass may and may not claim on that basis.

**The headline: the mechanism replicates in two genuinely new domains, and the
paper's central claim survives the out-of-fold correction. Two results came out
weaker or more complicated than hoped and are recorded honestly below.**

---

## Task 1 — Two new rare-event domains

Both datasets were verified publicly accessible and licence-compatible before
any work was built on them:

- **Healthcare** — UCI "Diabetes 130-US Hospitals" (id 296), 101,766 encounters,
  no credentialing. Downloaded via the UCI API (`/api/dataset?id=296`); note the
  older static `.zip` URL now 404s.
- **Climate** — NOAA Storm Events Database, yearly detail files 1996–2023
  (28 years). U.S. federal government work, public domain.

MIMIC was not pursued, per the spec's instruction not to wait on credentialing.

### The design trap that had to be fixed first

The first healthcare implementation regressed the raw 0/1 readmission indicator
scaled to 0–100. That construction is **degenerate for this paper's question**:
a rare unit's nonconformity score is then |p − 100| ≈ 100 by arithmetic, so
"more rare units in the calibration set" and "wider score support" become the
same variable. Measured collinearity was Spearman ρ = 0.68–0.72, and rare-count
appeared to *beat* diversity (ρ 0.94 vs 0.73) purely as an artifact.

The recession testbed never had this problem because its target is a continuous
probability whose residuals vary within rare months. Both new domains were
therefore built on **continuous rate/intensity targets**:

- healthcare: per-cohort 30-day readmission **rate** (cohorts of ≥25 encounters)
- climate: per-region-month significant-event **intensity**

This is a correction of the analysis design, not a search for a friendlier
result — the binary version cannot separate the two competing predictors even
in principle, whatever answer it returns.

### Result — the mechanism replicates

`fix-reg/table_crossdomain.csv`, `table_crossdomain.md`

| Domain | Model | ρ(diversity) | ρ(rare-count) | ρ(rare \| diversity fixed) |
|---|---|---|---|---|
| Macro-recession (US, 6M) | stacking-chain | **0.913** | 0.098 | 0.045 |
| Macro-recession (US, 3M) | stacking-chain | **0.683** | 0.331 | 0.312 |
| Healthcare (readmission) | ridge | **0.687** | 0.491 | 0.184 |
| Healthcare (readmission) | gradboost | **0.632** | 0.340 | 0.063 |
| Climate (storm intensity) | ridge | **0.675** | 0.565 | 0.359 |
| Climate (storm intensity) | gradboost | **0.638** | 0.431 | 0.256 |

In every domain, under every model, support-width diversity is the stronger
predictor of ACI coverage, and rare-event count attenuates once diversity is
held fixed. The fixed-size ablations (varying *only* rare-event count at N=254)
reproduce the monotone coverage gain seen in Table 1 in both new domains.

**Honest caveat — the margin is narrower outside macroeconomics.** In the
recession testbed at 6M, diversity outruns rare-count by ~50× in R²
(0.85 vs 0.02). In the new domains the ratio is roughly 2–3× (e.g. climate
ridge: 0.461 vs 0.344), and rare-count retains a non-trivial within-tertile
correlation in climate (0.256–0.359). The claim that survives all three domains
is *"diversity dominates rare-count"*, *not* *"rare-count is irrelevant"*. The
near-total redundancy of rare-count is a US-6M-specific finding.

---

## Task 2 — Model-agnosticism, and the overdue probit comparison

Every domain was run with two underlying models from the start (ridge and
gradient boosting); the relationship holds under both, as the table above shows.

For the recession testbed, the probit comparison was finally run
(`fix-reg/task2_probit_strategies.csv`): out-of-fold probit nonconformity
scores, then the full strategy comparison (pooled/trailing ACI, Mondrian,
PID-conformal, EVT-tail, diversity-maximizing selector).

**The result is degenerate, and must not be reported as "the probit wins."**
The probit reaches 96.9–100% coverage at every horizon — but its intervals are
**40–335× wider than the actual spread of the test target**. In the 2020+ test
window the 6M target only spans 0–1.64 on a 0–100 scale, so a ~30-point interval
covers everything trivially. Mondrian, EVT and diversity-optimal variants exceed
100 points wide, i.e. wider than the entire plausible range of a probability.

Consequences for the write-up:

1. Report the probit with a **width/target-spread ratio** next to every coverage
   figure, or the number misleads. Coverage alone cannot distinguish a
   well-calibrated interval from a vacuous one.
2. The selector shows **+0.00pp** on probit scores at every horizon. This is
   *saturation, not failure*: there is no headroom above ~97–100%. It is not
   evidence against the mechanism, and must not be presented as such.
3. The reviewer's question is nonetheless answered: ACI and the selector *can*
   be applied to the probit's residuals, and were.

---

## Task 3 — In-sample vs out-of-fold (the most important correction)

`fix-reg/task3_insample_vs_oof.csv`

| Horizon | In-sample trailing → div-opt | Out-of-fold trailing → div-opt | OOF div-opt Wilson |
|---|---|---|---|
| Current | 89.23 → 92.31 (+3.08) | 93.85 → 93.85 (+0.00) | [85.22, 97.58] |
| 1M | 85.94 → 89.06 (+3.12) | 89.06 → 96.88 (+7.82) | [89.30, 99.14] |
| 3M | 85.48 → 90.32 (+4.84) | 90.32 → 95.16 (+4.84) | [86.71, 98.34] |
| **6M** | **67.80 → 81.36 (+13.56)** | **84.75 → 96.61 (+11.86)** | **[88.46, 99.07]** |

**The direction of the gain survives**, which is the answer to reviewer uQqS's
central worry: the 6M selector improvement is +13.56pp in-sample and +11.86pp
out-of-fold, so it is not an artifact of in-sample residuals.

**But the OOF levels must not be quoted without their intervals** — see the
reconciliation below, which is the controlling reading of this table. Every OOF
coverage figure here spans or nearly spans the 90% target, and the in-sample and
OOF arms come from two different models.

### Reconciliation: what the OOF result does and does not license

**This section is load-bearing for Task 5 and must be read before any prose is
written about the paper's six-month limitation.**

The OOF 6M diversity-optimal figure is 96.61% — numerically *above* the 90%
nominal target. Taken at face value that would invert the paper's central honest
limitation ("coverage remains below 90% under every strategy tested"), which
anchors the Abstract, Results, Limitations, Conclusion and the title's closing
"not yet reached". It must not be read that way. Two facts constrain it.

**1. At this sample size the result is not distinguishable from nominal.**
Wilson intervals for every OOF arm (n = 59 scored months at 6M: 65 test months
minus the six-month forward tail):

| Horizon | n | OOF trailing | Wilson | OOF div-opt | Wilson | block-bootstrap |
|---|---|---|---|---|---|---|
| Current | 65 | 93.85 | [85.22, 97.58] | 93.85 | [85.22, 97.58] | [90.77, 100.0] |
| 1M | 64 | 89.06 | [79.10, 94.60] | 96.88 | [89.30, 99.14] | [95.31, 100.0] |
| 3M | 62 | 90.32 | [80.45, 95.49] | 95.16 | [86.71, 98.34] | [91.94, 100.0] |
| **6M** | **59** | **84.75** | **[73.48, 91.76]** | **96.61** | **[88.46, 99.07]** | [89.83, 100.0] |

At 6M **both** intervals span 90%: the baseline ([73.48, 91.76]) is not clearly
below nominal, and the diversity-optimal result ([88.46, 99.07]) is not clearly
above it. A Wilson interval containing 90% is equivalent to failing to reject
H0: p = 0.90, so no separate binomial test is reported — it would restate the
same fact.

The honest claim is therefore **not** "the six-month problem is smaller than we
thought", and emphatically **not** "the six-month problem is solved". It is:
*at this sample size we cannot tell whether six-month coverage clears or falls
short of 90% under honest scoring.*

**2. The in-sample and OOF numbers are not the same measurement corrected.**
The saved stacking ensemble was fit on all 635 training months and therefore
cannot produce honest out-of-fold scores. The OOF arm uses a **surrogate**
RegressorChain of the same architecture family, refit per rolling-origin fold.
So 67.80 → 81.36 and 84.75 → 96.61 come from *two different models*, and the
paper may not present the second as the first "corrected".

This confound also rules out a paired test that would otherwise be natural here.
A McNemar test on the two arms' covered/missed patterns over the same months
would conflate **scoring method** (in-sample vs out-of-fold) with **model
identity** (published vs surrogate), and could not isolate in-sample bias. It is
deliberately not reported.

**Suggested wording for the paper** (safe against both failure modes):

> Under out-of-fold scoring using a like-for-like reconstruction of the
> RegressorChain architecture (necessarily refit per fold, not the identical
> published model), the estimated 6-month baseline coverage rises to 84.75% and
> the diversity-optimal selector reaches 96.61% — both numerically closer to or
> above the 90% target than the in-sample figures suggested. However, at n = 59
> test months, Wilson intervals for both estimates span 90% ([73.48, 91.76] and
> [88.46, 99.07] respectively), so this cannot be read as resolving the coverage
> shortfall — only as evidence that the in-sample estimate may have overstated
> it. We report both the in-sample and out-of-fold results and flag this as an
> open question requiring a larger test window to resolve.

**Consequence for the title and closing line, flagged early.** Both are built on
the in-sample framing being unambiguous ("at six months it still falls short").
The honest picture is now *uncertain, possibly overstated* rather than *clearly
short*. This need not break the framing — "not yet reached" can legitimately mean
"not yet **confirmed** reached" rather than "confirmed absent" — but Task 5 must
navigate the distinction deliberately rather than swapping numbers into the
existing sentences.

---

## Task 4 — Confidence intervals

`fix-reg/task4_coverage_intervals.csv` — 61 coverage percentages across all
existing and new tables now carry Wilson intervals; the new domains and the
headline comparisons additionally carry moving-block bootstrap intervals
(block = 12 months, for temporal autocorrelation).

**This materially constrains what can be claimed.** With only 65 test months, a
Wilson interval is ±8.5pp wide on average:

- 6M headline: 67.80 [55.61, 77.80] → 81.36 [70.45, 89.11]. The intervals still
  **overlap slightly**, so this is strong suggestive evidence, not a decisive
  separation. Describe it as such.
- Current/1M/3M gains (+3.08, +3.12, +4.84pp) are **well inside the noise** and
  must not be described as improvements without this caveat.

The new domains are better powered (146 and 243 test units) and their intervals
are correspondingly tighter.

---

## Task 5/6 — What the writing pass may now claim

**Supported by the evidence produced:**

- The calibration-diversity finding holds across three genuinely distinct
  rare-event domains (macroeconomic, healthcare, climate) and is not
  macroeconomics-specific.
- It is model-agnostic: it holds under stacking chains, ridge, and gradient
  boosting.
- The *direction and magnitude of the selector's gain* survive out-of-fold
  scoring (+11.86pp at 6M), the correct rebuttal to the in-sample critique.

**NOT supported — do not write these:**

- "Rare-event count is irrelevant." True at US-6M; only *attenuated* elsewhere.
- "The selector improves coverage at all horizons." Only 6M (and OOF 1M/3M)
  clears the confidence intervals.
- "The probit is outperformed." The probit comparison is degenerate on this test
  window; report the width ratios instead of a winner.
- Any claim of general applicability to uncertainty quantification *broadly*.
  Three domains support "across rare-event prediction domains" — that is the
  ceiling this evidence buys, and the spec's own warning against scoping the
  claim above the evidence applies with full force here.
- **"Out-of-fold scoring shows six-month coverage reaches/exceeds nominal."**
  The 96.61% figure's Wilson interval spans 90%, and it comes from a surrogate
  model rather than the published one. See the Task 3 reconciliation — this is
  the single easiest sentence to get wrong in the rewrite.

Task 6's glossary pass is unaffected by these results and can proceed as
specified.

---

## Reproduction

```
python fix-reg/domain_healthcare.py        # Domain 2
python fix-reg/domain_climate.py           # Domain 3 (builds/caches NOAA panel)
python fix-reg/task_oof_and_probit.py      # Tasks 2 and 3
python fix-reg/make_crossdomain_table.py   # Task 1 deliverable table
python fix-reg/task4_confidence_intervals.py
```

Shared pipeline: `fix-reg/domain_common.py` (ACI runner, diversity statistics,
ablation, 200-draw sweep, Wilson/block-bootstrap intervals) — every domain uses
the same machinery as the recession testbed rather than a reimplementation.

Data: `data/domains/uci_diabetes_130.csv`, `data/domains/noaa/*.csv.gz`
(263MB raw; cached panel at `data/domains/noaa_panel.csv`).
