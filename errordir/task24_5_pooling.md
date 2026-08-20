# Task 24, Item 5 — Should Climate's Result Be Pooled with Recession's 6M?

**Condition met:** Item 4 found a real result in climate — win rate >50% against
7 of 8 baselines, non-vacuous (ratio 1.08), permutation-significant (BH q ≤
0.0027). So this item runs.

**Recommendation: NO. Do not pool. The climate result does not need pooling, and
pooling would damage it.**

## Why pooling is not motivated here

Item 5 exists to ask whether pooling buys *genuine additional statistical power*.
Applying that test:

**1. Climate is already adequately powered.** n=243 with BH q ≤ 0.0027 against
every baseline. Pooling is a remedy for underpowered results; this one is not
underpowered. The motivation Item 5 was written to guard against — reaching for
aggregation *because* individual results were weak — is precisely what pooling
would be doing here for recession, not for climate.

**2. The two results point in different directions.** Recession/6M (Task 23) was
*not* significant: 45.8% win rate against DtACI, BH q = 0.468, n=59. Climate is
significant with win rates of 52.7–89.7%. Pooling a clear positive with a clear
null does not produce a stronger positive; it produces a diluted estimate whose
interpretation depends entirely on the weighting, and invites the reading that
the combined result "supports" a recession finding the recession data does not
support on its own.

**3. The units are not commensurable.** Recession/6M is a monthly macro forecast
with target spread 0.596 and Winkler values around 116. Climate is a region-month
storm-intensity panel with spread 8.778 and Winkler around 11 — an order of
magnitude apart. A pooled Winkler mean would be dominated by recession's scale
regardless of which domain carried the effect. Any honest pooling would need
standardized per-domain scores, at which point it is a meta-analysis of two
studies, n=2, not a larger sample.

**4. The paper's own §4.2 precedent applies directly.** Extending the recession
window added only calm months, not new signal. Pooling across domains risks the
same failure in a subtler form: more rows, but the additional rows come from a
domain where the effect was absent, so they add noise to the estimate rather than
resolution.

## What is actually worth doing instead

The climate result stands on its own and does not need reinforcement. The
productive follow-ups are:

- **Replicate climate within itself.** 243 test region-months across 5 regions
  and many years — a genuine held-out split (e.g. by region, or a later time
  block) would test whether the effect is stable *inside* the domain where it
  was found. That is real additional evidence; pooling is not.
- **Resolve the gradboost gate failure.** Both gradboost fits failed Check 2 for
  a mechanical reason (the random-direction null is 3× higher for step-function
  models). Climate/gradboost had the *highest* held-out correlation of all four
  fits (0.374) and was never tested. A perturbation check valid for non-smooth
  predictors could double the evidence base without any pooling.
- **Close the per-axis gap.** Climate errordir wins on combined Winkler but
  loses coverage to diversity_optimal and width to bellman_ci. That is the
  specific, well-posed gap between "best tradeoff" and "beats current methods."

## Guardrail check

The guardrail forbids attempting this item if Item 4 found nothing. Item 4 found
something in climate, so evaluating pooling was legitimate. The evaluation's
conclusion is that pooling is not warranted — recorded here as the answer to the
question, not as a deferral.
