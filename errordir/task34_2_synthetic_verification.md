# Task 34, Item 2 — Synthetic Verification, Both Directions

## Result: **all three cases correct. DS detects known signal (percentile 1.000) and rejects known null in both the normal and hyper-responsive regimes (0.639, 0.463).**

Script: `task34_2_synthetic_verification.py` · Data: `task34_2_synthetic.csv`

## The cases

| case | requirement | responsiveness | DS | **DS percentile** | verdict | correct |
|---|---|---|---|---|---|---|
| A — known signal | DETECT | 0.027 | **+0.4765** | **1.000** | DETECT | **Yes** |
| B — known null | REJECT | 0.080 | +0.0047 | 0.639 | REJECT | **Yes** |
| C — known null, hyper-responsive | REJECT | 0.194 | +0.0128 | 0.463 | REJECT | **Yes** |

Case C is the one that matters most: a deeper, larger LightGBM (depth 8, 400
trees) with **7× the responsiveness of case A** — the exact condition Task 33
found breaks Check 2. DS still correctly rejects, at 0.463.

Both directions were required and both were run. A test verified only on
"can it find real signal" would be half-verified, and the failure mode it would
hide — passing everything — is worse than the strictness it replaces.

## The finding this verification produced

**Check 2 scores 0.000 on all three cases, including the known-signal case A.**

That is not a subtlety; it is a direct demonstration that Check 2 fails to detect
a difficulty direction that is present *by construction*, on tree-like data. The
candidate direction in case A is the literal generating direction of the
heteroscedasticity, and Check 2 ranks it below **every** random direction.

The mechanism is exactly Task 33's diagnosis. In case A the true difficulty
direction `w` is orthogonal to the directions driving the conditional *mean*
(`X[:,1..3]`). Perturbing along `w` changes where the noise is large but barely
moves the prediction; random directions, which load on the mean-driving features,
move it more. Check 2 measures magnitude, so it ranks the informative direction
last.

**This retroactively reframes the 0/20 tree record from Tasks 32–33.** Those
fits failed a test that provably cannot detect a real signal under these
conditions. The failures were never evidence that tree models lack a difficulty
direction — an interpretation Task 33 had already begun to suspect and this
verification now establishes on ground truth.

## Honest limits of the verification

- Case A's signal is strong and low-dimensional (d=8, one clean direction). Real
  fits have 12–21 features and any real signal is weaker; passing here does not
  guarantee power at realistic effect sizes.
- The null cases use a fixed arbitrary direction, since no true direction exists
  to supply. That is the correct construction, but it means the rejection is
  tested against one arbitrary direction rather than an adversarially chosen one.
- **Matched-direction counts vary by case** (200/200, 108/200, 190/200). Case B
  drops 46% of null directions as unmatchable within the [0.1, 10] rescale
  bound. The percentile there is computed over 108 directions, which is adequate
  but worth noting — a case where very few directions matched would give an
  unreliable percentile, and Item 3 reports this count for every fit.
