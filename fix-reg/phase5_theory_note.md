# Phase 5 — Light theoretical grounding

A single, precise, citable proposition connecting calibration-set nonconformity-
score breadth to ACI's coverage deficit, in the Gibbs & Candès (2021) framework.
It is deliberately small; the assumptions it needs (and where they fail in
practice) are stated explicitly.

## Setup & notation

Split/adaptive conformal with absolute-residual nonconformity score
`S = |Y − Ŷ|`. Calibration scores `{S_i}_{i∈C}` with empirical CDF `F̂_C` and
quantile `Q̂_C(1−α) = inf{s : F̂_C(s) ≥ 1−α}`. Test scores drawn from `G` (the
shifted test-time score law). ACI (Gibbs–Candès) adapts `α_t` by
`α_{t+1} = α_t + γ(α − 𝟙[miss_t])`; in the small-γ / stationary regime `α_t → α`
and the method reduces to fixed-quantile split conformal with threshold
`Q̂_C(1−α)`.

## Proposition (coverage deficit = quantile shortfall)

*In the fixed-quantile limit, the marginal test coverage of the ACI interval
`Ŷ ± Q̂_C(1−α)` is*
> `Cov = G(Q̂_C(1−α))`,
*so the coverage deficit relative to the nominal `1−α` is*
> `Δ = (1−α) − G(Q̂_C(1−α)) = G(Q_G(1−α)) − G(Q̂_C(1−α))`,
*which is non-negative and non-decreasing in the **quantile shortfall**
`δ := Q_G(1−α) − Q̂_C(1−α)` whenever `G` is continuous and increasing near
`Q_G(1−α)`. In particular `Δ = 0` iff `Q̂_C(1−α) ≥ Q_G(1−α)`: coverage is
achieved exactly when the calibration set's upper quantile reaches the test
set's.*

**Proof sketch.** Coverage of a fixed half-width `q` is `P(S_test ≤ q) = G(q)` by
definition of the score. Substituting `q = Q̂_C(1−α)` gives the first line. The
nominal target satisfies `1−α = G(Q_G(1−α))` by definition of `Q_G`. Subtracting
gives `Δ = G(Q_G(1−α)) − G(Q̂_C(1−α))`; since `G` is monotone, `Δ ≥ 0` and is
monotone in `δ`. The ACI wrapper adds an `O(γ)` drift term (Gibbs–Candès Thm 1
bounds the long-run miss rate at `α + O(γ/T)`), which does not change the
sign/monotonicity conclusion. ∎

## Why this makes "diversity governs coverage" precise

The deficit depends only on **where the calibration set's `(1−α)` quantile sits**.
A calibration set whose score distribution is *narrow* (small upper quantile) has
a large shortfall `δ` and large deficit; *broadening* the calibration score
distribution so its upper tail reaches `Q_G(1−α)` drives `δ → 0`. "Diversity"
(support width p95−p5, the Phase-1 statistic) is a proxy for exactly this upper-
quantile reach. Rare-event months help **only** insofar as they raise `Q̂_C(1−α)`
— which is why, once the quantile (diversity) is fixed, rare-event *count* is
redundant (Phase 1 matched-pairs).

## Numerical instantiation (6M, trailing N=254)

`Q̂_C(0.90) = 3.01` vs. `Q_G(0.90) = 27.06` → shortfall `δ = 24.0` pp →
predicted coverage `G(3.01) = 57.6%`, deficit **32.4 pp** — matching the observed
6M behavior. This is the mechanism behind every empirical result in Phases 1–4.

## Assumptions & where they fail in practice (stated honestly)

1. **Fixed-quantile limit.** Real ACI has finite γ; the `O(γ/T)` drift is small
   here (T=62) but nonzero — the proposition is about the limiting behavior, and
   finite-γ ACI can partially self-correct (seen mildly in Phase 3c PID).
2. **Marginal, not conditional.** It bounds marginal coverage; it says nothing
   about conditional coverage within the rare-event regime (where deficits are
   larger).
3. **Stationary `G` over the test window.** The 2020–2025 test window is itself
   non-stationary (COVID + tightening); `G` is an approximation.
4. **Breadth ⇏ correct quantile.** Crucially, raising *support width* helps only
   if it raises the *`(1−α)` quantile specifically*. A large-N diverse set can
   still dilute extremes: e.g. the fixed-N=254 all-rare set lifts `Q̂_C(0.90)`
   only 3.01→3.24 (coverage still 57.6%), whereas Phase 2's greedy extreme-tail
   selector raises it enough to reach 81% at 6M. So the operational lever is
   *upper-quantile reach*, of which support width is a necessary but not always
   sufficient proxy — consistent with Phase 2's "necessary but not sufficient"
   verdict.

**Status:** this is a ceiling-raiser, kept to one proposition + sketch as scoped.
It formalizes the empirics without overclaiming; the caveats (esp. #4) are
themselves informative and align with the Phase 2 finding.
