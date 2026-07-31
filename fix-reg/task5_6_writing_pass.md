# Tasks 5 & 6 — Writing Pass

**Status: the manuscript source is not in this repository.** A full-tree search
found no `.tex`, `.docx`, or `.odt` file, and no Abstract/Introduction/Conclusion
prose anywhere outside the analysis reports. The only occurrences of the paper's
framing language ("still falls short", "not yet reached") are in the analysis
memos, not in a manuscript.

So Tasks 5 and 6 are delivered here as **drop-in replacement prose plus an
applied-edit list**, ready to paste into the manuscript wherever it lives. What
cannot be done from this repo is the mechanical edit of the paper's own
Abstract/Intro/Conclusion text — that needs the source file. Everything that does
not require it is complete below.

The title/closing-line rewrite (§4) is flagged for review before it goes wide,
per the standing instruction.

---

## §1 — Task 5: the contribution hierarchy, stated explicitly

Reviewer uQqS's critique was that the paper's contribution hierarchy is unclear.
The resolution, to be stated plainly in both the Introduction and Conclusion:

> **The diagnostic finding is the paper's primary contribution.** Every domain,
> selector result, baseline comparison, and theoretical proposition in this paper
> exists to support and stress-test a single claim: that calibration-set
> diversity — the support width of the nonconformity-score distribution — rather
> than rare-event count, governs Adaptive Conformal Inference coverage under data
> scarcity. Recession forecasting is the primary deployment testbed, not the
> subject of the paper.

## §2 — Task 5: replacement Abstract

Drop-in, calibrated to what the evidence now supports:

> Adaptive Conformal Inference (ACI) is widely used to maintain coverage under
> distribution shift, but its behaviour when the calibration set contains few
> rare events is poorly characterised. We show that ACI coverage under rare-event
> scarcity is governed by the **diversity** of the calibration set's
> nonconformity scores — specifically their support width — and not by the count
> of rare events it contains. We demonstrate this across three genuinely distinct
> rare-event domains: macroeconomic recession forecasting (our primary deployment
> testbed), hospital readmission prediction, and extreme-weather event intensity.
> In every domain, and under every underlying prediction model tested (stacking
> chains, ridge regression, gradient boosting), support-width diversity is the
> stronger predictor of coverage, and rare-event count becomes largely redundant
> once diversity is held fixed. Building on the diagnostic, we introduce a
> diversity-maximizing calibration selector that improves coverage where
> size- and composition-based strategies do not, and we position the finding
> against the standard conformal toolbox: Mondrian, PID-conformal and
> extreme-value variants all fail to close the gap. We report confidence
> intervals on every coverage estimate and quantify the effect of in-sample
> versus out-of-fold scoring, which materially changes the interpretation of
> long-horizon results.

**Do not** add "for uncertainty quantification broadly" or similar. Three
rare-event domains buy "across rare-event prediction domains" and no more.

## §3 — Task 5: replacement Results/Generalization framing

The five-country table (Table 2) is demoted, not deleted:

> We first establish the mechanism on U.S. recession forecasting (§Table 1), then
> test its generality along two axes. **Within** the macroeconomic domain, five
> further national recession series replicate the relationship (§Table 2). Across
> **domains**, healthcare readmission and extreme-weather intensity replicate it
> again (§Table N), which is the stronger of the two checks: it is the one a
> reader cannot dismiss as macroeconomics-specific.

Honest qualifier that must accompany the cross-domain table:

> The margin is narrower outside macroeconomics. In the recession testbed at the
> six-month horizon, support-width diversity outpredicts rare-event count by
> roughly fifty times in R² (0.85 vs 0.02); in the new domains the ratio is
> nearer two- to three-fold, and rare-event count retains a non-trivial
> within-tertile correlation in the climate data (ρ = 0.26–0.36). The claim
> supported across all three domains is that **diversity dominates rare-event
> count**, not that rare-event count is irrelevant — the latter is a finding
> specific to the U.S. six-month horizon.

## §4 — Task 5: the title and closing line — FLAGGED FOR REVIEW

**This is the piece to review before it goes wide.** The existing title and
closing line assert that six-month coverage falls short of nominal. Under honest
out-of-fold scoring that assertion is no longer established — but neither is its
opposite. See the reconciliation in `multidomain_findings.md`.

The distinction that governs the rewrite: **"not yet reached" must become "not
yet confirmed reached", not "reached".** The evidence supports neither a claim of
success nor a continued claim of failure.

**Closing line — recommended replacement:**

> Whether six-month coverage can be brought to nominal remains open. Under
> in-sample scoring it clearly falls short; under out-of-fold scoring with a
> like-for-like model reconstruction the point estimates rise above the target,
> but at n = 59 test months the confidence intervals span it in both directions.
> What the evidence does establish is the mechanism: coverage tracks calibration
> diversity, in every domain we tested and under every model we tried.

**Title — three options, in order of preference:**

1. **Keep the existing title unchanged.** If it says "not yet reached" or
   similar, that phrasing survives honestly under the "not yet *confirmed*"
   reading. This is the lowest-risk option and my recommendation: the title makes
   a claim about the state of knowledge, which is still accurate.
2. **Shift the title onto the diagnostic**, which is the primary contribution and
   is not in doubt — e.g. framing around "calibration diversity, not rare-event
   count, governs adaptive conformal coverage". Better reflects the revised
   contribution hierarchy (§1); costs the recession-specific hook.
3. **Retain a shortfall claim explicitly.** Not recommended — it asserts more than
   the out-of-fold evidence now supports.

I cannot apply this without the manuscript, and would not want to regardless
until the choice is confirmed.

## §5 — Task 5: applied-edit list for the rest of the manuscript

Mechanical changes, once the source is available:

- Any sentence quoting **67.8% → 81.4%** without qualification must gain either
  "(in-sample scoring)" or the out-of-fold companion figures.
- Any sentence stating that six-month coverage **remains below nominal** must be
  softened per §4.
- Every coverage percentage in every table gains its Wilson interval — 61 are
  precomputed in `fix-reg/task4_coverage_intervals.csv`.
- The Current/1M/3M selector gains (+3.08, +3.12, +4.84pp) must not be described
  as improvements without noting they sit inside the ±8.5pp interval at n = 65.
- The probit comparison must report **width/target-spread ratios** (40–335×)
  alongside coverage, per the degenerate-result finding.
- `fix-reg/phase2_diversity_selection_method.md` closes by noting "a deployment
  version would need out-of-fold calibration scores". That is now done (Task 3);
  the sentence should point to the out-of-fold results rather than describe them
  as future work.

## §6 — Task 6: accessibility glossing pass

Plain-language definitions to add at first use. Written for the general AI
audience the paper now targets, one sentence each.

- **Support width** — how far apart the smallest and largest prediction errors in
  the calibration set are (measured as the 95th minus the 5th percentile, so a
  single outlier cannot dominate). Wide support means the calibration set has
  seen both calm and turbulent behaviour.
- **Nonconformity score** — how badly the model missed on one example; here, the
  absolute difference between prediction and outcome. Conformal methods turn a
  distribution of these past misses into a prediction interval for future ones.
- **Mondrian calibration** — calibrating each class or regime separately (e.g.
  recessions apart from expansions) instead of pooling all examples together.
- **PID-conformal** — a variant that adjusts the interval width using a
  control-theory feedback rule, reacting to both current and accumulated
  coverage error, like a thermostat correcting for persistent drift.
- **Quantile reach** — whether the calibration set's largest errors are big
  enough to match the errors actually encountered at test time. A calibration set
  can be diverse yet still fail to reach far enough into the tail.
- **Spearman ρ** — a correlation measure based on rank order rather than raw
  values, so it captures "when this goes up, does that go up?" without assuming a
  straight-line relationship.
- **Focal loss** — a training loss that down-weights easy, abundant examples so
  the model pays proportionally more attention to rare ones.
- **Diebold-Mariano test** — a statistical test for whether one forecaster's
  errors are genuinely smaller than another's, rather than smaller by chance.
- **Adaptive Conformal Inference (ACI)** — a method that widens or narrows its
  prediction intervals over time in response to whether recent intervals actually
  contained the outcome, targeting a chosen coverage rate under distribution
  shift.
- **Wilson score interval** — a confidence interval for a percentage that stays
  well-behaved with small samples and near 0% or 100%, where the textbook normal
  approximation misleads.
- **Block bootstrap** — resampling contiguous blocks rather than individual
  points, so that autocorrelation within a time series is preserved in the
  resampled data.

---

## What remains blocked

Applying §2–§5 to the manuscript itself. Provide the source file (or its path)
and these become mechanical edits. The prose above is written to be pasted
directly, and §4 is the piece to review first.
