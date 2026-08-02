# Task 14 — Compute Cost of Each Calibration Strategy

**Headline: the diversity-optimal selector is among the cheapest strategies
tested — 1.0× the baseline per step, with a one-off selection sweep of 0.06ms.
It is not buying its coverage with compute. The genuinely expensive options are
EVT-tail (110–199× per step) and, in a different way, CPTC.**

Task 9 gave a coverage-vs-sharpness tradeoff. This adds the third axis a
practitioner needs.

Script: `fix-reg/task14_compute_cost.py`. Data: `task14_compute_cost.csv`.
Method: median of 5 repeats, identical scores and test stream per domain.

---

## Per-step cost, relative to pooled/trailing ACI

| Strategy | Climate | Healthcare | Recession-6M |
|---|---|---|---|
| pooled/trailing ACI | 1.0× | 1.0× | 1.0× |
| **diversity-optimal (ours)** | **1.0×** | **1.0×** | **1.0×** |
| augmented trailing | 1.0× | 1.0× | 1.0× |
| PID-conformal | 1.0× | 1.0× | 1.0× |
| Mondrian | 1.0× | 1.0× | 1.1× |
| AcMCP | 1.4× | 1.4× | 1.3× |
| DtACI | 1.7× | 1.8× | 1.7× |
| Bellman CI | 2.6× | 2.8× | 3.2× |
| **EVT-tail** | **109.5×** | **158.4×** | **199.1×** |

Absolute per-step costs are ~20µs for the cheap strategies, so on these stream
lengths (65–243 points) everything except EVT-tail completes in single-digit
milliseconds. The relative column is what generalises to longer streams.

## One-off setup costs

| Strategy | Setup | What it is |
|---|---|---|
| diversity-optimal | **0.06ms** | one two-pointer sweep over the pool |
| augmented trailing | 2.3–4.0ms | GPD fit + synthetic draws |
| CPTC (HMM state input) | **190ms** | fitting the latent-state model |
| all others | 0 | no setup phase |

## The three findings worth stating in the paper

**1. The selector is free.** Its entire cost is a 0.06ms sweep, after which it
runs ordinary ACI at exactly baseline per-step cost. Whatever else the
coverage-for-width tradeoff costs, it does not cost compute. This is worth one
sentence because a reader might reasonably assume a "selection" method carries
overhead.

**2. EVT-tail is two orders of magnitude more expensive, and structurally so.**
It refits a generalized-Pareto MLE at *every* step (3.9ms per interval at
recession-6M). That is not an implementation inefficiency to optimise away —
re-estimating the tail per step is what the method does. Given that Phase 3b
already found EVT-tail *hurts* coverage (66% at 6M), it is now the worst
strategy on both axes tested: most expensive and among the least accurate.

**3. Pool augmentation inherits the selector's efficiency.** Its 2.3–4.0ms
setup buys a permanently baseline-cost stream, because the synthetic scores are
generated once and then consumed by ordinary ACI. So the Task 11 efficiency
result (most of the selector's coverage gain at a third of the width) comes with
no compute penalty either.

**CPTC is the interesting middle case.** Its per-step conformal machinery was
not benchmarked here because it needs the state matrix as an input, but its
190ms one-off HMM fit already exceeds every other strategy's total runtime. And
that is the *cheap* substitute — the authors' REDSDS is a deep switching-dynamics
model whose training cost is orders of magnitude higher and is **not**
represented in this table. For a practitioner, CPTC's real cost is "train and
maintain a second model", which is a different category of expense from
"microseconds per interval".

## Honest caveats

- **Wall-clock timings on one machine, single-threaded, medians of 5.** They are
  meaningful as *ratios* between strategies measured under identical conditions,
  not as absolute performance claims.
- **Stream lengths here are short** (65–243 points). Per-step costs are the
  figure that extrapolates; total runtimes do not.
- **CPTC's per-step cost is absent**, not zero. Only its state-model fit is
  measured. The comparison to other strategies' per-step numbers is therefore
  incomplete for that one method, and the table says so rather than implying a
  favourable gap.
- **The REDSDS cost is unmeasured and unmeasurable here** — no trainer was
  available (Task 7). The 190ms HMM figure is a floor on CPTC's true setup cost,
  not an estimate of it.
- **Implementation-dependent.** DtACI's 1.7× reflects 7 experts; a larger expert
  set would cost proportionally more. Bellman CI's 2.6–3.2× reflects a 50-point
  α grid and a 3-step lookahead. Both scale with parameters a user could change.
