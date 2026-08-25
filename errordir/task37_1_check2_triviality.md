# Task 37, Item 1 — Is Check 2's Ridge Pass Construction-Trivial?

## Result: **NO — the hypothesis is falsified. Fitted-but-meaningless directions score 0.18–0.48 and pass 0 of 20 times. Real β scores 0.995 on both domains. Check 2's ridge pass carries genuine information.**

Script: `task37_1_check2_triviality.py` · Data: `task37_1_check2_triviality.csv`

Check 2 implementation copied unmodified from `task34_3_retest.py` — this is a
diagnostic on the existing instrument, not a redesign.

## The comparison

Four candidate directions, all scored by the identical Check 2:

| domain | candidate | n | Check 2 mean | min | max | **pass rate** |
|---|---|---|---|---|---|---|
| insurance | **β (real)** | 1 | **0.995** | 0.995 | 0.995 | **1.00** |
| insurance | noise_target (fitted) | 10 | 0.395 | 0.085 | 0.840 | **0.00** |
| insurance | perm_target (fitted) | 10 | 0.409 | 0.015 | 0.920 | **0.00** |
| insurance | random_unit (untrained) | 10 | 0.525 | 0.000 | 0.975 | 0.10 |
| energy | **β (real)** | 1 | **0.995** | 0.995 | 0.995 | **1.00** |
| energy | noise_target (fitted) | 10 | 0.182 | 0.000 | 0.670 | **0.00** |
| energy | perm_target (fitted) | 10 | 0.482 | 0.085 | 0.800 | **0.00** |
| energy | random_unit (untrained) | 10 | 0.533 | 0.000 | 1.000 | 0.10 |

`noise_target` = RidgeCV(features → N(0,1)); `perm_target` = RidgeCV(features →
shuffled |error|). Both use the **identical fitting procedure** as β — same
estimator, same alphas, same features, same normalization — differing only in
whether the target carries error information.

## What this establishes

The hypothesis was that any regression-fit direction is geometrically favored on
a linear model, so Check 2 would pass for fitted-but-meaningless directions too.
**It does not.** The permutation control is the decisive one: `perm_target` uses
the *exact same error values* as β, merely shuffled, so it is fitted to a target
with identical marginal distribution and zero relationship to the features. It
averages 0.409 / 0.482 and never passes.

So Check 2 on ridge is not measuring "was this direction fitted." Something about
β's *specific alignment with where errors are large* is producing the 0.995.

Two honest qualifications:

1. **`random_unit` passes 10% of the time** (1 of 10 in each domain, max 0.975
   and 1.000). At a 0.95 threshold a random direction should pass ~5% of the
   time by definition, so 10% across n=10 is within sampling noise of nominal —
   but it does mean an individual Check 2 pass is not overwhelming evidence on
   its own. β at 0.995 sits well above that band.
2. **β is deterministic given the fit**, so it contributes n=1 per domain. The
   comparison is one real direction against 30 controls per domain, not a
   two-sample test with matched n. The separation (0.995 vs 0.18–0.53) is large
   enough that this is not the binding limitation, but it is worth stating.

## Consequence for Item 5

**No superseded banners are warranted.** The premise of Item 5 — "if Item 1 finds
Check 2 was also construction-trivial on ridge" — is not met. Tasks 21, 24–27 and
30 cited ridge Check 2 passes as meaningful evidence, and that citation stands.

This also sharpens what Task 36 corrected. Task 36 withdrew the claim that
"Check 2 certifies magnitude alignment, not difficulty relevance." That
withdrawal was correct *as a withdrawal* — the evidence Task 34 offered for it
was void. Item 1 now goes further and shows the claim was substantively wrong
too: Check 2 on ridge does discriminate difficulty-relevant directions from
fitted-but-irrelevant ones.
