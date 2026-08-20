# Task 20, Item 3 — Falsification: Do Either Signal Predict Misses?

**Result: NULL. Neither signal is validated. Nothing survives multiplicity
correction — minimum BH q-value 0.264 across the 24-test family, zero cells at
q < 0.10. Signal A additionally fails on sign: it is negative-lift in 5 of 8
cells, i.e. low-margin months missed *less* often than average.**

Scripts: `task20_3_validation.py`, `task20_3_leakcheck.py`
Data: `task20_3_validation.csv`, `task20_3_permutation.csv`, `task20_3_traces.csv`

## Method

Ran on the primary recession testbed's **existing** out-of-fold results —
`oof_pred` / `preds_test` / `y_test` imported unchanged from
`task_oof_and_probit.py`, the same import path `task7_baseline_horse_race.py`
uses. No new model runs. Both calibration strategies (pooled/trailing and
diversity-optimal), all four horizons.

The ACI loop was re-run step-by-step, recording at each test month the state
existing **before** that month's outcome was known (alpha_t, q_t, Q_C, Signal A,
Signal B), and only afterward reading whether the interval missed. Each signal
was tested **separately** — no combined score was constructed or evaluated,
per the guardrail.

## Results — miss rate in risk-flagged quartile vs. base rate

Signal A flags the *bottom* quartile (low margin = strained); Signal B the *top*
(high drift = risky). `BH_q` is Benjamini-Hochberg across all 24 tests;
`perm_p` is a circular-shift permutation null preserving temporal
autocorrelation.

| Horizon | Strategy | Signal | base% | flag% | lift_pp | fisher_p | BH_q | perm_p |
|---|---|---|---|---|---|---|---|---|
| Current | pooled | A_margin | 6.15 | 0.00 | **-6.15** | 1.0000 | 1.0000 | 1.0000 |
| Current | divopt | A_margin | 6.15 | 0.00 | **-6.15** | 1.0000 | 1.0000 | 1.0000 |
| 1M | pooled | A_margin | 10.94 | 13.04 | +2.11 | 0.4921 | 1.0000 | 0.3815 |
| 1M | pooled | B_drift_max | 1.72 | 6.67 | +4.94 | 0.2586 | 0.6206 | 0.2190 |
| 1M | divopt | A_margin | 3.12 | 0.00 | **-3.12** | 1.0000 | 1.0000 | 1.0000 |
| 3M | pooled | A_margin | 9.68 | 17.65 | +7.97 | 0.2000 | 0.6000 | 0.3235 |
| 3M | pooled | B_drift_max | 3.57 | 14.29 | +10.71 | 0.0591 | 0.2640 | 0.2125 |
| 3M | divopt | A_margin | 4.84 | 0.00 | **-4.84** | 1.0000 | 1.0000 | 1.0000 |
| 6M | pooled | A_margin | 15.25 | 26.32 | +11.06 | 0.1093 | 0.3747 | 0.1775 |
| 6M | pooled | B_drift_max | 11.32 | 28.57 | **+17.25** | **0.0358** | 0.2640 | 0.1080 |
| 6M | divopt | A_margin | 3.39 | 3.70 | +0.31 | 0.7101 | 1.0000 | 0.4610 |
| 6M | divopt | B_drift_max | 3.77 | 14.29 | +10.51 | 0.0660 | 0.2640 | 0.2130 |

(p90 variants track max almost exactly; full 24 rows in the CSV.)

## Reading this honestly

**The one apparently-significant cell is a multiplicity artefact.** 6M pooled
Signal B at nominal p=0.0358 is the headline candidate. But 24 tests were run,
and one p≈0.04 among 24 is roughly what chance produces. Under BH it becomes
q=0.264. Under a permutation null that preserves the signal's autocorrelation it
becomes 0.108. Reporting it as validated would be the exact error this item
exists to prevent.

**Signal B is directionally consistent but underpowered.** Every cell where
misses exist shows positive lift, and the effect grows with horizon
(+4.94 → +10.71 → +17.25 pp). That is the pattern a real effect would produce,
and it is *not* nothing. But with n=53 and a base miss rate of 11.3%, the 6M
pooled test has roughly 6 misses total in the flagged quartile. The paper's own
6M sample-size problem (n≈60) applies with full force here, and per the
guardrail it is stated rather than papered over: **this data cannot distinguish
a real moderate effect from noise.** Signal B is *not validated*; it is also not
refuted.

**Signal A is uninformative on this testbed, for a structural reason.**
*(Originally written as "refuted"; revised after the climate follow-up —
see `task20_5_climate.md`. On climate, where the margin reaches 2.31 instead of
bottoming out at 4.10, Signal A's lift flips positive on both models. The
recession result reflects this testbed's narrow alpha range, not a refutation of
the signal.)* Its lift is
*negative* in 5 of 8 cells — low-margin months missed less often than average,
the opposite of the predicted direction. The cause is visible in Item 1's
boundary table: in live operation ACI's alpha stays in a narrow band, so the
observed margin never leaves **4.10-6.89**. The signal is only informative near
`q -> 0.5` (margin -> 1), and ACI never goes there. Signal A is a correct
instrument pointed at a condition that does not arise. It is uninformative *as a live diagnostic on this
testbed*, not shown to be wrong as mathematics — and the climate follow-up
supports that reading.

**A structural note on the diversity-optimal strategy.** Its miss counts are
tiny (2-3 misses across a whole horizon; 0 misses at Current and 1M), which is
why several cells are exactly 0.00 with p=1.0. Those cells carry essentially no
information — there is nothing for a signal to predict. This is the
low-headroom problem from Task 19 Item 3, appearing again: the better-calibrated
the strategy, the less signal there is to validate against.

## No-leakage audit (`task20_3_leakcheck.py`)

The task's single hard constraint. Three independent checks:

1. **Positive control** — an oracle signal built from the outcome itself
   registers **+44.75 pp lift**. The quartile machinery detects real signal when
   one exists, so the null above is a property of the signals, not a broken
   pipeline. *(This check initially failed at +0.00 because a binary oracle makes
   `quantile(0.75) = 0`, flagging every month; jittering it to a continuous score
   fixed the control, not the pipeline.)*
2. **Shift invariance** — corrupting all realized scores at steps `>= t` leaves
   Signal B bit-identical (`1.726358` both ways). No forward reach.
3. **Q_G never used** — absent from executable code in both the signal module
   and Item 3. The only occurrences anywhere are prose statements saying it is
   not used.

## Verdict

Neither signal is validated. Per the guardrail, this is reported as a complete
answer, not quietly dropped: **Item 4 (packaging) does not run.**
