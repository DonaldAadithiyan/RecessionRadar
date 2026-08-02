# Task 16 — Summary: Four Deeper Analyses on the Quantile-Reach Mechanism

**Two items strengthen the paper, two require framing changes — and one of those
two is a result the paper currently gets backwards.**

| Item | Question | Outcome | Framing changes? |
|---|---|---|---|
| **A** | Does support width predict Q_C, or only coverage? | **Both pre-registered expectations held.** Q_C beats support width in 7/8 cells; supp→Q_C median ρ = 0.662 | **No** — vindicates §4, states it more precisely |
| **B** | Multivariate decomposition, N fixed | Support width dominates, but **rare-count's CI excludes zero in 5/7 cells** | **Yes, mildly** — adopt "dominates, not irrelevant" |
| **C** | Synthetic phase diagram | **The synthetic model mis-predicts the sign at every real domain's coordinates** | **Yes, materially** — §5.2 cannot be cited as corroboration |
| **D** | Other target coverage levels | Holds at 90–95% nominal, decays and inverts by 75–80% | **Yes** — add a scope statement |

---

## Item A — the theory section is vindicated

The chain `support width → Q_C → coverage` was argued analytically and checked
at one point. It is now measured at all three links.

- **Q_C predicts coverage better than support width in 7 of 8 cells**, with the
  margin growing at harder horizons (6M: **ρ = 0.961** vs 0.725).
- **Support width predicts Q_C at ρ = 0.622–0.741** across all three domains — a
  good but imperfect proxy, exactly as §4 claims.

Roughly a quarter of the predictive relationship is lost by describing the
mechanism in terms of support width instead of Q_C. That is the price of the
proxy, and it is worth stating. It does not change the practical
recommendation: Task 10 showed the selector attains maximal Q_C anyway, so
optimising the proxy and the target coincide.

→ `task16a_qc_direct_test.md`

## Item B — "dominates", not "redundant"

With both predictors and N fixed in one model:

- **Recession 3M/6M behave as the paper claims**: rare-count's CI includes zero,
  partial R² = 0.004 and 0.011 against support width's 0.385 and 0.492.
- **Healthcare and climate do not**: rare-count's CI excludes zero in all four
  cells, partial R² 0.040–0.117, and in **climate/ridge it is the larger
  coefficient** (0.368 vs 0.329).

This is sharper than the within-tertile check, which averaged the two questions
together. The paper's Task 1 scoping — *diversity dominates rare-count; it is
not that rare-count is irrelevant* — is now quantified with intervals. Near-total
redundancy is a US-recession-long-horizon result, not a general one.

→ `task16b_multivariate.md`

## Item C — the synthetic section fails its own validation

**This is the item that changes the paper most, and not in the intended
direction.**

Mapping the full frequency × separation plane (66 cells, after fixing a
saturation artifact in the spec's suggested grid) shows a clean monotone decay
in Δ_S crossing zero around Δ_S ≈ 1.25–2.0. Then:

| Domain | Δ_S | Synthetic predicts | **Reality measures** |
|---|---|---|---|
| Recession 6M | 1.26 | ≈ −0.02 | **+0.442** |
| Recession 3M | 2.19 | ≈ −0.16 | **+0.402** |
| Healthcare | 2.41 | ≈ −0.24 | **+0.112 / +0.132** |
| Climate | 2.97 | ≈ −0.36 | **+0.005 / +0.118** |

**All four real domains sit in the synthetic map's negative region, and all four
measure positive gaps.** The simulation gets the sign wrong at every real
coordinate.

This does **not** weaken the empirical finding — real measurements stand on
their own, and diversity dominates in 12/12 temporal cells (Task 13) and 7/8
model cells (Task 16A). It **does** mean §5.2 cannot be presented as
corroborating evidence: mapped properly, the same construction contradicts the
real results. The likely cause is that the two-component independent-draw
generator makes rare-count artificially informative — knowing the rare count
nearly determines the upper tail, which is untrue of real correlated,
continuum-valued scores.

**Recommendation:** report the grid as a validation check the synthetic model
failed, keep the figure (it shows precisely where simulation and reality part
company), and stop citing seven synthetic scenarios as support.

→ `task16c_phase_diagram.md`

## Item D — a scope statement, and it is the one the theory predicts

Sweeping α ∈ {0.05 … 0.25} on the primary testbed:

- **6M holds throughout** (+0.216 to +0.442).
- **3M and 1M decay**; 1M crosses zero near α ≈ 0.16 and inverts to −0.138.
- **The gap peaks inside ACI's measured operating range** [0.073, 0.132].

The finding is a property of the **high-coverage regime (90–95% nominal)** — the
regime conformal prediction is deployed in. At 80% nominal the operative
quantile is the 80th percentile, which is body rather than tail, so diversity of
the extremes stops mattering and count-based composition does as well.

**The decay is what §4 predicts**, and saying so converts an apparent limitation
into a confirmation of the mechanism. That connection should go in the paper.

→ `task16d_alpha_sweep.md`

---

## Pipeline change this task required

All four items needed per-draw data the sweep discarded.
`domain_common.random_draw_sweep` now records **Q_C** (the calibration set's own
(1−α) quantile) and accepts **`alpha_target`**. Both additive; every existing
caller verified unaffected. Raw per-draw recession data is exported to
`task16_perdraw_recession.csv` for audit.

Without this the spec's guardrail would have bound — none of the four could have
been done honestly from reported aggregates.

## What the paper should do

1. **§4 (theory):** state the chain as measured, including that support width
   loses ~25% of the relationship versus Q_C, and add Item D's connection —
   the mechanism weakens as the operating quantile leaves the tail, as predicted.
2. **§5.2 (synthetic):** demote from corroboration to a failed validation check,
   with the phase diagram shown and the mismatch explained.
3. **Redundancy claims:** adopt "diversity dominates rare-count" with Item B's
   intervals; reserve "redundant" for US recession at 3M/6M.
4. **Add a scope sentence:** the finding is established for 90–95% nominal
   coverage and decays below that.
