# Task 34, Item 1 — Test Design: Directional Specificity

**Stated in full before application to any real fit.** This is instrument
design; nothing below is adjusted after seeing which version makes trees pass.

## The problem being fixed

Check 2 asks: *does the candidate direction's raw induced-change magnitude
exceed the 95th percentile of a random-direction null?*

On trees that question is nearly uninformative. Task 33 measured a
zero-response rate of 0.00003 and median relative change 2.7× ridge's; Task 25
measured gradboost's null p95 at 3× ridge's **with lower variance**. The null is
high **and tight** — every direction moves the output by a similarly large
amount. A percentile threshold against that distribution has almost no room for
a real signal to separate, whether or not one exists.

Critically, magnitude is the wrong quantity. "This direction moves the
prediction a lot" is true of nearly all directions on a tree. What should
matter is whether the induced change is **difficulty-relevant**.

## The new statistic: Directional Specificity (DS)

For a fit with standardized test points `x_1..x_m`, realized nonconformity
scores `e_1..e_m`, and a candidate direction field `v(x)` (a single global
vector, or per-point for locally-varying mechanisms):

**Step 1 — induced change per point.**

    delta_i(v) = | f(x_i + h*v_i) - f(x_i - h*v_i) |        h = 0.5 SD

**Step 2 — the specificity statistic.**

    DS(v) = Spearman( delta_i(v),  e_i )    over i = 1..m

This asks: *do the points this direction moves most also happen to be the points
with the largest realized error?* A direction that moves everything equally has
DS ≈ 0 regardless of how large the movement is. A direction that moves
hard points more than easy ones has DS > 0. **Magnitude drops out; only the
cross-point pattern survives.**

Spearman rather than Pearson: `delta` and `e` are both heavy-tailed, and only the
ordering is claimed to matter.

**Step 3 — the magnitude-matched null.**

This is the part that controls for generic responsiveness. For each of `N = 200`
random unit directions `u_j`:

    1. compute delta_i(u_j) for all i
    2. RESCALE u_j's step so its mean induced change matches the candidate's:
           s_j = mean_i delta_i(v) / mean_i delta_i(u_j)
           delta_i^matched(u_j) = delta_i(u_j * s_j)   [recomputed, not scaled]
    3. DS_j = Spearman( delta_i^matched(u_j), e_i )

The rescale is applied to the **step size** and the perturbation is
**recomputed**, not linearly rescaled — for a tree, scaling the output would be
wrong, since the response is not linear in step size. Matching is on the *mean
induced change across points*, so every null direction produces the same average
movement as the candidate. What differs between them is only how that movement is
*distributed across points*, which is exactly the quantity of interest.

Where a rescale would need `s_j` outside [0.1, 10] the direction is dropped as
unmatched (recorded); this bounds the search rather than extrapolating wildly.

**Step 4 — the test.**

    percentile = fraction of j with DS_j < DS(v)
    PASS iff percentile >= 0.95

Same 0.95 bar as Check 2 — the threshold is unchanged, only the statistic and the
null construction differ.

## Why this is sensitive where Check 2 is not

- On **ridge**, magnitude and specificity are strongly coupled (a linear model's
  response is `w·v` for every point, so a direction aligned with `w` moves
  everything proportionally). DS should behave similarly to Check 2 there, which
  is what the ridge control in Item 3 verifies.
- On **trees**, magnitude is nearly constant across directions but the
  *distribution* of movement across points need not be. DS can discriminate
  precisely in the regime where a magnitude percentile cannot.

## What this test can still fail to detect

Stated in advance so the result is interpretable either way:

1. If a tree's response is not only large but also **uniformly patterned** — the
   same points move most regardless of direction — then DS will be near-constant
   across the null too, and nothing will separate. That would be a genuine
   negative, and a more informative one than Check 2's.
2. DS uses realized test-period errors `e_i`. These are **outcomes**, so DS is a
   *validation diagnostic only* — it can never be used to construct an interval.
   This is the same status Check 2 has (it also uses held-out data), and it is
   why Item 3 reports it alongside, not instead of, the existing gate.

## Pre-registered synthetic verification (Item 2)

Both cases required; passing only the first is insufficient.

- **Known-signal:** tree-like data where difficulty genuinely varies along a
  known embedded direction. The test must **detect** it (percentile >= 0.95).
- **Known-null:** tree-like data with no difficulty direction, only noise. The
  test must **reject** it (percentile < 0.95). This is the check that stops the
  redesign from being merely lenient — a test that passes both is a worse
  instrument than the strict one it replaces.

A third check is added, not required by the task but cheap and directly relevant:
the known-null case is also run at **elevated responsiveness** (a tree fit that
moves a lot for every direction), confirming the null-rejection holds under
exactly the condition that broke Check 2.
