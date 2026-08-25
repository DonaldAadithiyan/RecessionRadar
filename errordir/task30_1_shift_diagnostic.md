# Task 30, Item 1 — Is Energy's Regime Change Covariate Shift or Concept Drift?

## Finding: **energy is covariate shift AND concept drift.** RLCP was therefore tested partly outside the scope of its stated guarantee. Claim 1's framing must be narrowed accordingly — stated here, before Item 2, per the guardrail.

Script: `task30_1_shift.py` · Data: `task30_1_shift.csv`

## Test 1 — does P(X) shift?

| domain | per-feature KS median D | features with p<0.01 | **FIT-vs-TEST classifier AUC** |
|---|---|---|---|
| energy | 0.213 | 92% of 12 | **0.799** (moderate) |
| climate | 0.173 | 62% of 21 | **0.958** (strong) |

Both domains have real covariate shift. **Climate's is substantially stronger** —
a classifier separates climate's FIT from TEST rows almost perfectly on features
alone.

## Test 2 — does P(error | z) hold? (matched z bins, defined on FIT)

Conditioning on z is what separates the two shift types: matched z means matched
predicted difficulty, so a remaining error difference is drift in the
*relationship*, not just the input distribution.

### Energy

| bin | n_fit | n_test | median &#124;e&#124; FIT | median &#124;e&#124; TEST | KS p |
|---|---|---|---|---|---|
| 1 | 347 | 110 | 0.404 | 0.564 | **0.0016** |
| 2 | 347 | 47 | 0.402 | 0.459 | 0.357 |
| 3 | 346 | 52 | 0.581 | 0.579 | 0.354 |
| 4 | 347 | 77 | 0.611 | 0.804 | 0.029 |
| 5 | 347 | 114 | 0.978 | **2.202** | **0.0000** |

**40% of bins drift significantly.** Bin 5 — the hardest, highest-z cases — shows
median error rising **2.25×** at matched difficulty. That is the relationship
itself moving, not the inputs.

### Climate (control)

| bin | median &#124;e&#124; FIT | median &#124;e&#124; TEST | KS p |
|---|---|---|---|
| 1 | 1.999 | 1.380 | 0.016 |
| 2 | 2.044 | 2.277 | 0.380 |
| 3 | 2.219 | 2.832 | 0.706 |
| 4 | 2.066 | 3.441 | 0.054 |
| 5 | 3.243 | 4.350 | 0.506 |

**0% of bins drift** at p<0.01.

## The control is what makes this conclusive

| domain | covariate shift | concept drift | did methods break? |
|---|---|---|---|
| climate | **0.958 (stronger)** | **0.00 of bins** | **no — everything ties** |
| energy | 0.799 (weaker) | **0.40 of bins** | **yes — RLCP/CQR/q_α all fail** |

Climate has *more* covariate shift than energy and nothing breaks. Energy has
*less* covariate shift but real concept drift, and three methods fail. The
diagnostic discriminates, and it points squarely at concept drift — not covariate
shift — as what breaks the frozen-calibration methods.

## How this changes claim 1, explicitly

**Previous framing (Tasks 27–29):** "RLCP, CQR and q_α(z) all fail on energy
under distribution shift, where errordir's ACI anchor adapts online." This
implied RLCP failing a guarantee it claims to provide.

**Corrected framing:** RLCP's guarantee covers *covariate shift*. Energy contains
covariate shift **and** concept drift, and the evidence points to the concept
drift as the operative cause — climate's stronger covariate shift breaks nothing.
**No covariate-shift guarantee promises robustness to concept drift, so RLCP's
energy failure is not a violation of its stated guarantee.**

This is a **narrowing, not a retraction**. What survives is still a real,
mechanistically-identified advantage, but a more specific one:

> Under **concept drift** — where the feature→error relationship itself moves —
> methods that freeze a calibration-period estimate (RLCP, CQR, q_α(z))
> degrade, while errordir's online ACI anchor adapts. This is a property no
> covariate-shift guarantee addresses, so it is a genuine gap in what current
> methods offer rather than a failure of them to deliver what they promise.

Two honest consequences:

1. **It is no longer "RLCP's guarantee breaks under real conditions."** It is
   "RLCP's guarantee does not extend to concept drift, and concept drift occurs
   in practice." Weaker as a criticism of RLCP, still useful as a scope statement.
2. **Claim 1 remains n=1.** Energy is the only domain in this project with
   detected concept drift. The mechanism is well-identified — three independent
   methods failing the same way, with a control domain that isolates the cause —
   but its generality is untested. This is what the noted Task 31 should address.

## Guardrail compliance

Diagnostic only. No intervals were constructed and Task 29's comparison was not
re-run.
