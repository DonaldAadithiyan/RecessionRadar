# All-Phases Summary — Strengthening the Calibration-Diversity Finding

**Central claim under test:** ACI coverage under rare-event temporal shift is
governed by the **breadth (diversity) of the calibration set's nonconformity-score
distribution** — not by calibration size, and not uniquely by rare-event count.

All phases completed. No forecasting models retrained (everything is
recombination/resampling of the saved ensemble's residuals, transparent AR(1)
baselines for the cross-country series, or synthetic scores). Per-phase detail
files linked below.

## Per-phase verdicts

- **Phase 1 — Quantify the relationship → GO.** [phase1_diversity_relationship.md](fix-reg/phase1_diversity_relationship.md).
  200 random N=254 subsets. **Support width (p95−p5)** is the best diversity
  statistic and dominates rare-event count as a coverage predictor: at 6M,
  support R²=0.85 vs rare-count R²=0.02 (50×); at 3M, 0.26 vs 0.11. Matched-pairs
  shows rare-count **redundant once diversity is fixed** (within-tertile ρ≈0).
  Honest caveat: Current/1M are saturated, so the relationship lives at 3M/6M.

- **Phase 2 — Diversity-aware selection method → works; partial 6M win.**
  [phase2_diversity_selection_method.md](fix-reg/phase2_diversity_selection_method.md).
  A greedy support-width-maximizing selector improves coverage at all 4 horizons
  and is the **only strategy in the paper to move 6M** (67.8→81.4%). Honest: it
  buys coverage with much wider intervals, 6M still < 90%, and diversity-optimal
  ≈ rare-heavy on this pool → "necessary but not sufficient at long horizons."

- **Phase 3a — Mondrian ACI → backfires.** [phase3_baseline_comparisons.md](fix-reg/phase3_baseline_comparisons.md).
  Class-conditional ACI *lowers* coverage everywhere (6M 67.8→61.0), because the
  test window is 63 expansion vs 2 rare months, so it stops borrowing wide
  rare-event scores. Even with oracle regime labels. Positions the paper
  **against** the textbook fix.

- **Phase 3b/3c — EVT-tail & PID-conformal → neither fixes it.** Same file.
  PID nudges 6M to 71% (still ~19pp short); EVT-tail *hurts* (66%). Confirms the
  deficit is a calibration-**diversity** problem, not a weak-base-algorithm
  problem — "you can't model your way to a tail your calibration set never saw."

- **Phase 4a — Synthetic generalization → holds.** [phase4a_synthetic_generalization.md](fix-reg/phase4a_synthetic_generalization.md).
  Across 7 synthetic variants (frequency/magnitude/clustering), diversity beats
  rare-count and matched-pairs holds everywhere. Honest nuance: the margin
  narrows at high frequency/magnitude (predictors become collinear) — so the
  claim is strongest in the real low-frequency macro regime.

- **Phase 4b — Cross-series → replicates on 5 countries.** [phase4b_cross_series_check.md](fix-reg/phase4b_cross_series_check.md).
  Euro area/UK/Germany/Japan/Canada OECD recession series (fetched from FRED,
  leakage-safe AR(1)): support-width diversity predicts coverage far above
  rare-count on every one; matched-pairs holds. Honest caveat: these are binary
  indicators smoothed to rolling rates (higher base rate than the US smoothed
  probability) — a conservative, imperfect analog, flagged as such.

- **Phase 5 — Theory note → one clean proposition.** [phase5_theory_note.md](fix-reg/phase5_theory_note.md).
  In the fixed-quantile limit of ACI (Gibbs–Candès), coverage deficit
  `Δ = G(Q_G(1−α)) − G(Q̂_C(1−α))` is monotone in the calibration→test
  **quantile shortfall**; Δ=0 iff the calibration upper quantile reaches the
  test's. Numerically instantiated at 6M (predicted 57.6% = observed). Key
  honest caveat (assumption #4): breadth helps only if it raises the *specific*
  (1−α) quantile — a large-N diverse set can dilute extremes, which is exactly
  why Phase 2's extreme-tail greedy selector was needed.

- **Phase 6 — Presentation pass.** Paper-ready summary figure
  `figures/methods_coverage_comparison.pdf` (coverage by horizon across pooled /
  Mondrian / PID / diversity-optimal; grayscale-safe, house style) plus the
  Phase-1 scatter `figures/phase1_diversity_vs_coverage.pdf`. Detailed
  figure/table/section restructuring intentionally deferred (as scoped) now that
  the evidence set is known.

## What changed about the paper's central claim

**Sharpened, not overturned.** Every phase supports "diversity, not size or
rare-event count" and makes it more rigorous:
- It is now a **continuous quantitative relationship** (Phase 1) with a
  **matched-pairs redundancy test**, not 5 points + 3 spot checks.
- It **generalizes** to synthetic conditions (4a) and 5 independent real series (4b).
- It is **positioned against the standard conformal toolbox** — Mondrian, PID,
  EVT-tail all fail to fix it (Phase 3), while a diversity-optimal selector
  helps (Phase 2).
- It has a **small theoretical backbone** (Phase 5).

**One complication, reported honestly (does not threaten the claim):** the
operative lever is the calibration set's **upper (1−α) quantile reach**, of which
support-width diversity is a strong but not perfect proxy — at large fixed N,
diverse-but-averaged sets can still under-reach the tail (Phase 5 #4, Phase 2
6M). So the sharpest statement is: *coverage is governed by whether the
calibration score distribution's upper tail reaches the test-time tail; rare-event
episodes and expansion-vintage breadth both help by widening that tail, and
rare-event count per se is redundant once tail-reach is fixed.*

**Unchanged / still open:** 6-month coverage cannot be brought fully to nominal
by any calibration strategy tested (best: 81% via diversity-optimal) — a genuine
limitation, honestly retained.

## File index
- Phase MDs: `fix-reg/phase1_diversity_relationship.md`, `phase2_diversity_selection_method.md`, `phase3_baseline_comparisons.md`, `phase4a_synthetic_generalization.md`, `phase4b_cross_series_check.md`, `phase5_theory_note.md`
- Data: `fix-reg/phase1_random_sweep.csv`, `phase1_correlations.csv`, `phase1_matched_pairs.csv`, `phase2_selection_comparison.csv`, `phase3a_mondrian.csv`, `phase3bc_extra_baselines.csv`, `phase4a_synthetic_generalization.csv`, `phase4b_cross_series.csv`
- Figures: `figures/phase1_diversity_vs_coverage.{pdf,png}`, `figures/methods_coverage_comparison.{pdf,png}`
- Scripts: `fix-reg/task_phase1.py`, `task_phase2.py`, `task_phase3a.py`, `task_phase3bc.py`, `task_phase4a.py`, `make_fig_phase1.py`, `make_fig_phase6.py`
