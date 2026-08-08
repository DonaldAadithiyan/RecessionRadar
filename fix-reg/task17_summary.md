# Task 17 — Summary: Five Experiments Closing the Paper's Weak Points

**Three items strengthen the paper, one corrects a stated claim, and one
weakens a headline number. All five changed something.**

| Item | Question | Result | Changes a claim? |
|---|---|---|---|
| **3** | Verify the q ≥ 0.5 optimality boundary | Ratio exactly 1.0 for α < 0.5, breaks **at** 0.5 | **Yes** — boundary must be **strict**: q > 0.5 |
| **1** | Bootstrap the headline R² | 6M decisive (ΔR² 0.392 [0.167, 0.574]); **3M crosses zero** | **Yes** — don't claim 3M; requote 6M with CI |
| **2** | Does generic spreading substitute for the extremes? | No — stratified captures **14–20%**. And coverage tracks Q_C **exactly** across 5 rules | **Yes, favourably** — selector vindicated, but not unique |
| **5** | Should Q_C replace support width as primary vocabulary? | Yes (10:1 partial-R² asymmetry) — but support width keeps a small **real** contribution | **Yes** — §4 leads with Q_C |
| **4** | Correlated generator (allowed to fail) | **Succeeded, 4/4 correct signs** vs 0/4 for the independent one | **Yes** — §5.2 restorable as corroboration |

---

## Item 3 — the boundary is off by its endpoint

Ceiling ratio is exactly **1.00000 for every α ≤ 0.49**, then **0.518 at exactly
α = 0.500**, then collapses beyond. Four violations, all at the single point
α = 0.5.

Not a bug: at N = 254 the selector supplies 127 extreme scores and the α = 0.5
quantile needs exactly 127. **The inequality becomes tight, and tight is not
strict.** The proof's mechanism is confirmed precisely as argued; only its
boundary phrasing is wrong.

**Correction: §4.3 should read q > 0.5 (α < 0.5), not q ≥ 0.5.** No published
number changes — ACI operates at [0.073, 0.132], with enormous margin — but the
stated claim is wrong at its endpoint and a reader checking the algebra would
find it.

→ `task17_3_alpha_boundary.md`

## Item 1 — the headline needs an interval, and 3M needs a retraction

| Horizon | ΔR² median | 95% interval | P(Δ>0) | Excludes 0 |
|---|---|---|---|---|
| 3M | 0.205 | **[−0.012, 0.396]** | 0.960 | **No** |
| 6M | 0.392 | **[0.167, 0.574]** | 1.000 | **Yes** |

B = 1000, block-resampling the **pool** (not the draws, which would double-count
their dependence on it).

**6M survives decisively.** **3M does not** — its lower bound is a hair below
zero, so it is directionally consistent (96% of replicates) but not established.

The intervals are wide, which is itself the finding the spec asked to be
reported plainly. And the published **0.85 vs 0.02 sits well outside** this
out-of-fold interval (0.448 vs 0.046) — consistent with Task 3's in-sample
optimism result, but it means that number should never be quoted unlabelled.

→ `task17_1_bootstrap.md`

## Item 2 — the reviewer's alternative fails, and the mechanism is confirmed by a new route

Stratified sampling captures only **19.9%** (decile) and **14.2%** (quintile) of
the selector's gain, reaching Q_C ratios of ~0.56. Spreading across all deciles
wastes most of the budget below the operative quantile.

**The deeper finding:** coverage tracks Q_C exactly across five structurally
different rules. `tail_plus_recency` shares only **65.4% of indices** with the
selector yet produces **identical Q_C (68.7195) and identical coverage** —
verified directly because exact agreement looked like a bug.

So the selector is vindicated against the obvious alternative, **but it is not
unique**: any rule attaining the same Q_C performs identically. It is *a* way to
reach maximal tail-reach at fixed N, not *the* way. Practical corollary: half the
budget can go to recency without cost, provided the other half reaches the tail.

→ `task17_2_stratified_baselines.md`

## Item 5 — lead with Q_C, but don't call support width a pure proxy

| | 3M | 6M |
|---|---|---|
| partial R² of **Q_C** given support | 0.557 | 0.602 |
| partial R² of **support** given Q_C | 0.053 | 0.105 |

A ten-to-one asymmetry. Q_C alone explains 73–79% of coverage variance; adding
support width buys 1.5–2.2 points. **§4 should lead with Q_C** and describe
support width as the observable, pre-calibration diagnostic.

But β(support)'s CI **excludes zero at both horizons**, so support width carries
something real beyond its own quantile — the spec's "more surprising" outcome, in
mild form. Plausibly it is denoising: support width aggregates two order
statistics where Q_C is one, and is more stable at N = 254. This analysis cannot
separate that from a conceptually distinct contribution, and says so.

→ `task17_5_qc_vs_support.md`

## Item 4 — the diagnosis was right

| Domain | Independent generator (Task 16C) | **Correlated generator** | Actual |
|---|---|---|---|
| Recession 3M | −0.16 ✗ | **+0.389** ✓ | +0.402 |
| Recession 6M | −0.02 ✗ | **+0.269** ✓ | +0.442 |
| Healthcare | −0.24 ✗ | **+0.421** ✓ | +0.122 |
| Climate | −0.36 ✗ | **+0.439** ✓ | +0.062 |

**4 of 4 correct signs, against 0 of 4.** Coupling event probability and score
magnitude through a shared latent severity was the missing ingredient, exactly
as Task 16C diagnosed. Parameters were fixed in advance and the generator was
run once, per the guardrail against tuning.

Two honesty notes carried into the write-up: it **over-predicts magnitude 3–7×**
in healthcare and climate (qualitative corroboration only), and one grid column
(Δ_S = 0.75) is a **fixed-test-seed artifact**, disclosed rather than smoothed.

→ `task17_4_correlated_synthetic.md`

---

## What the paper should do

1. **§4.3:** change q ≥ 0.5 to **q > 0.5**. One character, but it is currently wrong.
2. **§4 framing:** lead with **Q_C**; demote support width to the observable
   diagnostic, while noting its small reliable independent contribution.
3. **Headline number:** quote **ΔR² = 0.392, 95% CI [0.167, 0.574]** at 6M.
   **Drop the 3M separation claim.** Never quote 0.85 vs 0.02 unlabelled.
4. **Selector motivation:** add the stratified comparison — it defeats the
   obvious alternative — and state that the selector is one route to maximal
   tail-reach rather than the only one.
5. **§5.2:** restore as corroboration using the **correlated** generator, and
   report the independent generator's failure as the contrast that identified
   the missing ingredient.
