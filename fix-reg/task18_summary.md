# Task 18 — Summary: Full Review Response

**All nine coding items attempted and completed, including the two the spec
marked optional. Three results are unfavourable to the paper and are reported as
such. Two answer reviewer questions whose premises turned out to be false.**

| Item | Result | Changes a claim? |
|---|---|---|
| **1** Regime-weighted CP | **Matches the selector's coverage at 3M/6M** — the first standard-toolbox method to do so — at ~14% more width | **Yes** — add to baseline table as the strongest comparator |
| **2** Pooling | **No pooling occurs.** Selection is per-horizon; both comparisons N/A | **Yes** — state it explicitly in the method section |
| **3** Winkler score | **DtACI beats the selector at all 4 horizons.** So does the trailing baseline | **Yes** — cannot claim utility-optimality |
| **4** EVT threshold | **The GPD fit is unstable** (ξ flips sign in every horizon, +1.61 to −1.93) | **Yes** — EVT's failure is partly a fitting artefact |
| **5** Recency/tail split | **Width is NOT constant** across splits; and any tail share ≥25% is equivalent | **Yes** — corrects the review's premise; useful deployment finding |
| **6** UP-OCP | Undercovers badly (77.97–81.54%); loses to trailing at all 4 | No — new baseline row |
| **7** Quantile convention | **Conventions agree exactly** for all α < 0.5 (32/32 each) | No — confirms the proof |
| **8** Generator magnitude | **Neither modification helps**; both worse than baseline | No — characterises a known limitation |
| **9** Algorithm box | Pseudocode extracted; odd-N and tie behaviour documented | **Yes** — paper needs the box, with edge cases |

---

## The three unfavourable results

**Item 3 — the selector loses on a proper scoring rule.** Winkler score (fixed
before computing, per the guardrail):

| Horizon | Selector | Trailing | **DtACI** |
|---|---|---|---|
| Current | 147.20 | 118.72 | **111.05** |
| 1M | 108.55 | 95.52 | **76.84** |
| 3M | 139.40 | 118.64 | **109.74** |
| 6M | 148.62 | 122.77 | **116.71** |

The selector is the *worst* of the three. This is consistent with everything
already established (Task 9's sharpness finding, Task 11's augmentation
motivation) but it is the first time it has been priced on one number. The paper
can claim coverage-optimality; it cannot claim utility-optimality.

**Item 4 — EVT-tail was never given a stable fit.** ξ flips sign in all four
horizons across thresholds, and 6M coverage moves 8.5 points (83.0 → 91.5) on
threshold choice alone. The paper's "EVT-tail hurts" conclusion holds for the
q = 0.80 configuration but should be reported as configuration-specific rather
than a verdict on extreme-value calibration.

**Item 1 — a standard-toolbox method finally matches the selector.**
Regime-weighted CP closes the gap completely at 3M and 6M (95.16 and 96.61,
identical to the selector) and overshoots at 1M. It costs ~14% more width and
requires a test-time regime label the selector does not need — but it belongs in
the table, and omitting it would be the gap the reviewer suspected.

## The two false premises

**Item 2 — there is no pooling.** Selection is per-horizon and per-model
throughout; verified in both `task7_baseline_horse_race.py` and
`task_phase2.py`, then confirmed empirically (the four horizons select sets
overlapping only 52–69%; pooling would give 100%). The reviewer's concern about
pooled selection destabilising ACI's per-horizon α_t updates does not arise.

**Item 5 — width is not constant across splits.** The review asked for the
recency comparison "holding width constant"; width actually jumps 2–5× between
0% and 25% tail allocation, then is exactly flat. The comparison as framed
cannot be made.

That flatness is itself the useful finding: **any tail allocation ≥25% gives
identical coverage, width and Q_C.** A practitioner needs only a quarter of the
budget in the tails; the rest is free. This is the third independent appearance
of the ceiling effect (Task 10, Task 17 Item 2, and now this).

## The confirmations

**Item 7** — nearest-rank and linear-interpolation quantile conventions agree
**exactly** for every α < 0.5 (32/32 each), disagreeing only at α = 0.5 itself,
where the underlying inequality goes tight. The proof's O(1/N) claim is confirmed
and in fact understated.

**Item 8** — neither heteroskedastic noise nor state-dependent coupling narrows
the generator's magnitude gap; both make it worse. Informative rather than null:
heteroskedastic noise nearly *perfectly* matches recession while worsening
healthcare and climate, which points at domain structure the generator omits
(competing predictors, low headroom) rather than a missing noise term.

**Item 6** — the parameter-free baseline undercovers at all four horizons because
its AdaGrad step (η ≈ 0.74 → 0.30) is ~60× ACI's γ and never decays over a
65-month stream. A small-sample failure, not a refutation of the approach.

**Item 9** — two edge cases the prose never covered: **odd N gives the extra
point to the low tail**, and ties break by NumPy's default (non-stable) argsort,
which diverges from stable sorting on tie-heavy data. Neither affects any
published number (all four score pools have zero duplicate values, and N = 254 is
even), but both belong in the algorithm box for reimplementation correctness.

## What the paper should do

1. **Add regime-weighted CP** to the main baseline table as the strongest
   comparator, noting it needs a test-time regime label the selector does not.
2. **Add the Winkler comparison** with the honest framing: the selector maximises
   coverage, DtACI wins on width-penalised utility.
3. **Report EVT-tail's threshold sensitivity** — one sentence plus the ξ table.
4. **State the per-horizon selection convention** explicitly in the methods.
5. **Replace "holding width constant"** with the split ablation's actual finding:
   ≥25% tail allocation suffices, and width is bimodal not constant.
6. **Include the algorithm box** with odd-N and tie-breaking specified.
7. Add one-line rows for UP-OCP and the quantile-convention check.

## Not addressed here, per the spec

- **Question 7 (code/data release)** — a policy statement for the rebuttal
  letter, not an experiment.
- **Table artifacts** — needs a visual check of the compiled PDF against the
  source CSVs, flagged separately.
