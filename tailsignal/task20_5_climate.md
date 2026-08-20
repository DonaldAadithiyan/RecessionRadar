# Task 20, Item 5 (follow-up) — Signal A on the Climate Domain

**Question asked: does climate's alpha genuinely approach the q=0.5 boundary,
turning Item 3's structurally-impossible test into a real one?**

**Answer: no — but the test was worth running, and it changed the conclusion
about Signal A in a way the recession testbed could not have shown.**

Script: `task20_5_climate.py` · Data: `task20_5_climate.csv`,
`task20_5_climate_alpha.csv`, `task20_5_climate_permutation.csv`,
`task20_5_climate_traces.csv`

## Step 1 — Climate does not reach the boundary either

Margin = 1.000 requires q = 0.500 (alpha = 0.500); margin < 2 ("strained")
requires alpha >= 0.250. Measured from climate's own full ACI traces (n=243
region-months, both models, both strategies):

| Model | Strategy | alpha range | q_min | distance to 0.5 | min margin |
|---|---|---|---|---|---|
| ridge | pooled | 0.096–0.123 | 0.8770 | +0.377 | 4.188 |
| ridge | divopt | 0.100–**0.216** | 0.7840 | +0.284 | **2.309** |
| gradboost | pooled | 0.100–0.132 | 0.8680 | +0.368 | 3.765 |
| gradboost | divopt | 0.100–**0.206** | 0.7940 | +0.294 | **2.396** |

The strict hypothesis is not met: nothing approaches margin → 1.

But climate is **not** simply a repeat of recession. Under the diversity-optimal
strategy climate's alpha reaches **0.216**, versus recession's maximum of 0.112
across all four horizons — roughly an **8× wider excursion**, pushing the margin
down to 2.31 against recession's floor of 4.10. Climate exercises meaningfully
more of the signal's dynamic range than any recession horizon does, even though
it stops short of the boundary.

## Step 2-3 — What the validation shows

The consequential finding is a **sign flip on the pooled baseline**:

| Model / strategy | Signal | n | base% | flag% | lift_pp | fisher_p | BH_q | perm_p |
|---|---|---|---|---|---|---|---|---|
| ridge/pooled | A_margin | 243 | 9.05 | 14.12 | **+5.06** | 0.0396 | 0.2376 | 0.1490 |
| gradboost/pooled | A_margin | 243 | 9.05 | 15.07 | **+6.01** | 0.0322 | 0.2376 | 0.0615 |
| ridge/pooled | B_drift_max | 237 | 9.28 | 11.67 | +2.38 | 0.3069 | 0.4604 | 0.3535 |
| gradboost/pooled | B_drift_max | 237 | 9.28 | 15.00 | +5.72 | 0.0699 | 0.2796 | 0.1390 |
| ridge/divopt | A_margin | 243 | 0.41 | 0.00 | -0.41 | 1.0000 | 1.0000 | 1.0000 |
| gradboost/divopt | A_margin | 243 | 1.23 | 3.12 | +1.89 | 0.1704 | 0.3408 | 0.2710 |

On recession, Signal A's lift was **negative in 5 of 8 cells**. On climate's
pooled baseline it is **positive in both models**, in the predicted direction,
at n=243 rather than n=59 — four times the sample size.

**It still does not survive correction.** Nominal p ≈ 0.032–0.040 becomes
BH q = 0.238, and the circular-shift permutation null (which preserves the
signal's autocorrelation) gives p = 0.062 and 0.149. Same pattern as Item 3: the
nominal significance is not robust.

## Step 4 — One confound ruled out

On the pooled baseline, supply is computed empirically and alpha_t moves with
recent misses, so low margin could be a proxy for "ACI recently raised alpha
because it was missing" — miss autocorrelation, not tail-reach diagnosis.
Tested directly:

- `rho(margin, prev_miss)` = **+0.105** (ridge), **+0.039** (gradboost) — weak.
- Lift persists after conditioning on the previous month's outcome:
  within `prev_miss=0`, lift = **+4.03 pp** (ridge, n=220) and **+6.63 pp**
  (gradboost, n=220).

So the positive lift is not an artefact of recent-miss autocorrelation. It is
simply too small relative to noise at this sample size to claim.

## What this changes

**Signal A should no longer be described as refuted.** Item 3's conclusion —
"negative lift in 5/8 cells" — was measured on a testbed where the margin never
left 4.10–6.89. On climate, where the margin reaches 2.31, the sign flips
positive on both models and survives a confound check. That is consistent with
the signal being real but only detectable once alpha has room to move, which is
exactly the hypothesis this follow-up was run to test.

It is **not** evidence that the signal works. Two positive cells at
perm_p = 0.062 and 0.149 are encouraging, not conclusive, and the
diversity-optimal cells (0.41–1.23% base miss rate — 1 to 3 misses in 243
months) again carry almost no information.

**The revised status of Signal A: not validated, not refuted — untested at the
condition it was designed for.** No domain in this project drives alpha near
0.5. Settling it requires a testbed whose ACI genuinely operates at low q, and
that does not exist here.

Signal B is unchanged by this follow-up: positive but non-significant
everywhere (best perm_p = 0.139).

## Item 4 status: still gated off

Nothing here validates a signal at the standard Item 3 set. The gate holds.
