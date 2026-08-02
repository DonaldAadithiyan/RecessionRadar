# Task 9 — Summary: Do the Three Open Threads Close?

Short answer: **two close, one narrows but stays open.** No result overturns any
existing claim in the paper; two results let existing caveats be dropped, and
one new tradeoff should be added.

| Part | Thread | Outcome | Does the paper's framing change? |
|---|---|---|---|
| **A** | Six-month coverage "undetermined" at n=59 | **Still undetermined at n=71.** One arm (in-sample baseline) resolves to "falls short". | **No** — keep the current language, update n and note that in-sample now resolves. |
| **B** | BCI reported untuned only | **Tuned, +3 to +23pp better. Still 0 of 8 vs the selector.** | **Yes, mildly** — drop the "not a ceiling" caveat; add the sharpness tradeoff. |
| **C** | CPTC never attempted | **Attempted (HMM variant). 0 of 8 vs the selector.** | **Yes** — "no baseline tested beats the selector" no longer needs the CPTC exception. |

---

## Part A — the six-month question does not resolve

Test window extended 65 → 79 months with real new FRED data; **n at six months
went 59 → 71**.

- **Out-of-fold diversity-optimal: 95.77%, Wilson [88.30, 98.55]** — lower bound
  still 1.7pp short of clearing 90%. Straddles.
- **Out-of-fold baseline: 85.92% [75.98, 92.17]** — straddles.
- **In-sample baseline: 81.69% [71.15, 88.98]** — **resolves**, entirely below
  90%. The paper's originally-reported deficit is now confirmed *under in-sample
  scoring*.

**Why more data did not settle it:** the 14 new months are a calm period — peak
recession probability 1.16% versus 100% in the original window. They add
easy-to-cover expansion months, not the rare-event transitions the six-month
deficit is actually about. A larger sample of the easy case cannot answer a
question about the hard one. At the observed rate, resolution needs n ≈ 100
(roughly late 2028).

**One caveat that must travel with these numbers:** the panel had to be rebuilt
from raw FRED series (STL is a global smoother and cannot be extended
incrementally). 13 of 46 features reconstruct at r < 0.8. A decomposition run
bounds the impact — reconstruction effect ≤1.4pp, new-data effect ≤0.4pp — so
the conclusions hold, but these are "recomputed on a rebuilt panel" numbers, not
a drop-in replacement for Table 8.

→ `task9a_extended_recession.md`

## Part B — tuned BCI is much better, and still loses

Tuning on a validation slice carved from the **calibration pool only** (test
data never touched; 60-config grid over λ_init, λ_max, γ):

- Gains of **+3.1 to +22.6pp**. Climate 78 → 91%, healthcare 66 → 88%. The
  paper's existing caveat that untuned BCI "should not be read as a ceiling" was
  correct and can now be replaced with the tuned numbers.
- **0 of 8 comparisons beat the selector.**
- **Recession 6M is the exception that proves the rule:** tuning cannot fix it
  (49 → 58%, still ~32pp short). An algorithm that optimises length subject to a
  coverage constraint cannot manufacture tail coverage the calibration scores
  never contained — exactly the paper's thesis.

**New tradeoff worth a sentence in §7:** tuned BCI hits ~91% in climate at
*half* the selector's width. The selector wins on coverage; BCI wins on
sharpness where it reaches nominal.

→ `task9b_bci_tuning.md`

## Part C — CPTC attempted, with a disclosed exception

**The one protocol exception in the whole evaluation:** CPTC requires a
state-probability matrix, so one latent-state model per domain was fit (training
data only) to supply it. The authors' REDSDS ships as precomputed inference
outputs, not a trainer, so a **Gaussian HMM** was used instead. Results are
labelled **"CPTC (HMM state input)"** throughout and are not the published
method's numbers.

- CPTC lands near nominal everywhere (86.5–91.8%) — a competent baseline.
- **0 of 8 beat the selector**, so the framing closes.
- **But at recession 6M it posts 91.84% at width 81.8, versus the selector's
  96.61% at 123.9** — nominal coverage at two-thirds the width, the best
  efficiency profile of any strategy at the paper's hardest horizon. Its
  interval straddles 90%, so this is not a solution to the six-month problem,
  but it should not be omitted.

→ `task9c_cptc.md`

---

## What the paper should now say

1. **Six-month framing: unchanged.** Still "undetermined under honest scoring",
   now at n=71. Add that the in-sample deficit is confirmed, and that resolution
   requires either ~29 more months or a test window containing an actual
   rare-event transition.
2. **"No baseline tested beats the diversity-optimal selector": now fully
   closed.** Nine strategies across three domains, including tuned BCI and a
   CPTC variant. The CPTC caveat can be dropped, replaced by one sentence
   disclosing the HMM substitution.
3. **Add the coverage-vs-sharpness tradeoff.** The selector maximises coverage
   and pays 1.2–2.6× in width. Tuned BCI and CPTC reach nominal in
   healthcare/climate at half to two-thirds that width. For a practitioner
   targeting "nominal, as tight as possible" rather than "maximum coverage",
   those are the better tools in those domains. This strengthens the paper's
   honesty without weakening its claim.
4. **Nothing requires retraction or revision.** All three parts confirm existing
   findings.

## Guardrail compliance

- Part A extended the window; Stage 1/Stage 2 were **not** retrained.
- Part C's HMM is the **only** model fit anywhere in Tasks 7–9, disclosed in the
  first paragraph of its write-up, its results table label, and here.
- Every coverage number in all three parts carries a Wilson interval.
- No result is called a win without CI separation — which is why B and C are
  both reported as 0 of 8 despite competitive point estimates.
