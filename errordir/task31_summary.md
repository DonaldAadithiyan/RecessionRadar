# Task 31 — Does the Adaptivity Finding Generalize Beyond Energy?

## **Outcome 2: NOT CONFIRMED.** Claim 3 does not generalize cleanly to a second concept-drift domain. Energy's result should be treated as direction-specific, not as a general property of frozen-estimate methods under drift.

## What was tested

Item 1 screened two candidates with `task30_1_shift.py`'s diagnostic applied
unchanged, before either touched the method.

| candidate | classifier AUC | **drifted z-bins** | err ratio | qualifies |
|---|---|---|---|---|
| fraud (credit-card) | 0.477 | **0.00** | 1.047 | **No** |
| **epidemic (OWID COVID)** | 0.992 | **1.00** | **0.443** | **Yes** |
| *energy (reference)* | *0.799* | *0.40* | *1.315* | *Yes* |
| *climate (control)* | *0.958* | *0.00* | *1.276* | *No* |

**Fraud was rejected for a concrete, reportable reason**: the OpenML copy has no
`Time` column, so I used row order as an arrival-order proxy and flagged in
advance that a failed proxy invalidates the screen. AUC 0.477 — below chance —
showed the halves are indistinguishable, so the rows are not time-ordered.
Adversarial drift in fraud is real; *this dataset* cannot express it.

**Epidemic qualified decisively**: 4 of 4 bins drift at p<0.001, the strongest
concept-drift signal in the project.

## The finding Item 1 surfaced before results existed

**Epidemic's drift runs opposite to energy's.** Errors *shrink* in the test
period (ratio 0.443) rather than grow (1.315). I flagged this in advance as
making epidemic a **stricter** test: energy's result could be explained trivially
by "any method that widens after misses does better when errors grow," whereas
epidemic requires the method to *narrow* to win.

That prediction proved exactly right, and it is what the task turned on.

## The result

| class | n | median Winkler | best |
|---|---|---|---|
| online | 9 | 2.016 | 1.434 (pooled_trailing) |
| frozen | 2 | 2.633 | 2.622 (cqr) |

**Mann-Whitney (online < frozen): p = 0.109 — not significant.**

Three things block a confirmation:

1. **Underpowered class test.** Only two frozen-estimate methods exist in the
   suite; the separation cannot reach significance.
2. **An online method is the worst in the table.** diversity_optimal (6.486) is
   2.5× worse than either frozen method. The online/frozen split is not the
   operative variable.
3. **errordir over-covers at 100.00%**, ranks 5th of 11, and loses significantly
   to four simpler online methods (win rate 4.8% vs pooled_trailing).

errordir *does* beat both frozen methods significantly (q=0.0000). But those wins
come from a method that is itself badly calibrated here, so they do not support
the mechanism claim.

## Why errordir failed in this direction

Its ACI anchor adapts by **widening after misses**. When errors shrink there are
almost no misses to trigger downward adaptation, and the multiplier's 0.75 floor
bounds how far it can narrow. It inherits calibration-period widths that are too
large — the mirror image of energy, where growing errors happened to reward its
widening behaviour.

**Claim 3, corrected:** errordir's online anchor is an advantage when drift makes
errors *grow*, and a liability when drift makes them *shrink*. Energy's result
was direction-specific, and the mechanism as previously stated ("online adapts,
frozen does not") is too general.

## Where the three-part claim now stands

| claim | status |
|---|---|
| **1. Ties on accuracy** | Confirmed (Task 29) — climate Winkler 11.220 vs RLCP 11.292, q=0.556. |
| **2. Wins on data efficiency** | Confirmed (Task 30) — 8×/10× slower degradation than RLCP, rising unbounded rate. |
| **3. Wins on adaptivity** | **Not confirmed.** Direction-specific; fails when drift reduces error magnitude. |

Claims 1 and 2 were not reopened here and are unaffected. Claim 3 should be
dropped from the headline or restated narrowly as a direction-dependent property
with n=1 supporting evidence.

This is the more useful outcome than a confirmation would have been: an
opposite-direction domain identified a real, specific failure mode — the
multiplier floor prevents adaptation downward — that a same-direction replication
would have missed entirely.

## Files

- `task31_build.py`, `task31_1_screen.py`, `task31_1_screening.csv`,
  `task31_1_domain_screening.md`
- `task31_23.py`, `task31_2_baselines.csv`
- `task31_3_comparison.csv` / `.md`, `task31_3_significance.csv`
