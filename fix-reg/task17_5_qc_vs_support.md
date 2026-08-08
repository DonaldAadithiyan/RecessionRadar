# Task 17 Item 5 — Should Q_C Replace Support Width as the Primary Vocabulary?

**Verdict: yes, lead with Q_C — but support width is not purely a proxy. Its
partial R² given Q_C is small (0.053 at 3M, 0.104 at 6M) yet its coefficient's
confidence interval excludes zero at both horizons. Support width tracks
something real beyond its own (1−α) quantile. That is the more surprising of the
two outcomes the spec described, and it earns its own sentence.**

Script: `fix-reg/task17_135_boundary_bootstrap_qc.py`.
Data: `task17_5_qc_vs_support.csv`.

Model, 200 draws per horizon, standardized:
`coverage ~ β₀ + β₁·Q_C + β₂·SupportWidth`

---

## Result

| Horizon | R²(Q_C alone) | R²(support alone) | R²(both) | β(Q_C) | β(support) | **partial R² of support given Q_C** | partial R² of Q_C given support |
|---|---|---|---|---|---|---|---|
| 3M | **0.729** | 0.422 | 0.744 | +0.750 [.656, .844] | +0.159 [.065, .253] | **0.053** | 0.557 |
| 6M | **0.792** | 0.532 | 0.814 | +0.745 [.660, .830] | +0.207 [.122, .292] | **0.105** | 0.602 |

## The asymmetry is decisive on the framing question

**Q_C given support width: partial R² = 0.557 / 0.602.**
**Support width given Q_C: partial R² = 0.053 / 0.105.**

Roughly a ten-to-one asymmetry. Q_C alone explains 73–79% of coverage variance;
support width alone explains 42–53%; adding support width to Q_C buys 1.5–2.2
percentage points of R² (0.729→0.744, 0.792→0.814).

**Section 4 should lead with Q_C** and describe support width as the observable,
pre-calibration diagnostic. That ordering follows from the numbers, not from
preference, exactly as the spec required.

## But support width is not *purely* a proxy, and that is worth reporting

The spec anticipated two outcomes: negligible partial contribution (confirming
pure-proxy status), or a real independent contribution (the surprising case).
**The result is the surprising case, in a mild form.**

β(support) is small but its CI excludes zero at both horizons — [0.065, 0.253]
at 3M and [0.122, 0.292] at 6M. Support width carries information about the
calibration set that its own 90th percentile does not.

A plausible reading: Q_C is a single order statistic and therefore noisy at
N = 254, while support width (p95 − p5) aggregates two order statistics and is
more stable. Support width may partly be *denoising* Q_C rather than measuring
something conceptually distinct. This analysis cannot separate those
explanations — it would need a variance decomposition across draw sizes — so the
honest statement is that the independent contribution is real but its
interpretation is open.

## Recommended framing for the paper

> The operative quantity is the calibration set's (1−α) quantile, Q_C: it alone
> explains 73–79% of coverage variance across random calibration draws, and
> support width adds only 1.5–2.2 points beyond it. Support width is the
> observable diagnostic — computable before any calibration is run, and
> model-free — and it remains the right statistic to *report*. Q_C is the right
> quantity to *reason with*. Support width does retain a small but statistically
> reliable independent contribution, which may reflect its greater stability as
> an aggregate of two order statistics rather than one.

This also aligns with Task 17 Item 2, where five structurally different
selection rules produced coverage that tracked Q_C exactly — including two rules
sharing only 65% of their members yet reaching identical Q_C and identical
coverage.

## Honest caveats

- **Q_C and support width are strongly correlated** (Task 16A: ρ = 0.62–0.74),
  so the partial R² split between them is sensitive to that collinearity.
  Inflated standard errors make the CIs conservative, which cuts *against*
  finding support width significant — so its independent contribution is not an
  artifact of collinearity.
- **Q_C is evaluated at the fixed nominal α = 0.10**, while ACI's α drifts during
  a run. A drift-aware Q_C might explain more and shrink support width's
  residual contribution further.
- **Recession testbed, 3M and 6M only**, per the spec. Healthcare and climate
  were not run; Task 16A's three-way comparison covers them at the correlation
  level and shows the same ordering.
- **200 draws, one seed.** The CIs describe variation across those draws, not
  across pools; Item 1's bootstrap shows pool-level uncertainty is considerably
  wider.
- **Linear and additive**, like every other regression in the paper.
