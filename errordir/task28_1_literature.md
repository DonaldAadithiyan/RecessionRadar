# Task 28, Item 1 — Literature Check: Does This Already Exist?

## Verdict: **yes, substantially. The core mechanism — conditioning conformal interval width on a learned low-dimensional representation — is established prior work. The specific scarcity-efficiency claim appears less directly tested, but adjacent results exist.**

This changes what Items 3–5 must be compared against, and it materially weakens
the "new method" framing.

## The closest existing method: RLCP (Randomly Localized Conformal Prediction)

Hore & Barber, *Conformal prediction with local weights: randomization enables
robust guarantees* (arXiv:2310.07850).

RLCP weights calibration conformity scores by kernel proximity to the test
point, `∝ exp(−γ‖X_i − X_{n+1}‖²)`. Critically for this task: **RLCP has a
documented low-rank projection variant that applies the Gaussian reweighting to
distances in a latent embedding space** rather than raw feature space.

That is the same structural idea as this proposal — conditioning width on a
learned low-dimensional coordinate rather than the full feature vector. And RLCP
carries theoretical guarantees this project's method does not:

| | RLCP | errordir (this project) |
|---|---|---|
| marginal validity | **proven** (via randomization) | not established |
| local coverage guarantee | **relaxed guarantee proven** | not established |
| validity under covariate shift | **proven** | not established — and Task 27 showed CQR failing exactly here |
| conditioning coordinate | kernel distance in latent embedding | 1-D projection β′x |

RLCP is strictly stronger on theory. errordir's differences are that its
coordinate is a *supervised, error-predicting* direction (β fit to predict
|error|) rather than an unsupervised embedding distance, and that it validates
that direction with an explicit two-part gate. Those are real differences, but
they are refinements within an occupied space, not a new space.

## Broader occupied territory

- **Localized Conformal Prediction** (Guan, *Biometrika* 110(1):33, 2023) — the
  general framework for proximity-weighted calibration residuals.
- **SpeedCP: Fast Kernel-based Conditional Conformal Prediction**
  (arXiv:2509.24100) — kernel conditional conformal, efficiency-focused.
- **Spatial Conformal Inference through Localized Quantile Regression**
  (arXiv:2412.01098) — localized quantile regression, close to this task's Item 3
  `q_α(z)` construction.
- **Beyond Marginal Validity: Finite-Sample Guarantees for Localized Conformal
  Prediction** (arXiv:2608.06206).
- **Enhanced localized conformal prediction with imperfect auxiliary
  information** (arXiv:2606.08551).

Item 3's proposed mechanism — a monotone, smoothed conditional quantile function
estimated on calibration data and indexed by a scalar coordinate — is very close
to localized quantile regression as already published.

## Is the scarcity-efficiency claim already tested?

Less directly, but the territory is not empty. Searches surfaced calibration
sample-efficiency results including a method reported to reach comparable
quantile-regression intervals with **20× fewer calibration samples**, and
conformal work operating with calibration sets as small as ~47 points. Those are
not framed as "low-dimensional conditioning is more sample-efficient than
raw-feature conditioning," which is this project's specific claim, but they show
sample-efficiency in conformal calibration is an actively worked question rather
than an unclaimed gap.

**I did not find a paper testing precisely this comparison**: a validated 1-D
supervised difficulty coordinate versus standard baselines, matched at equal
calibration sizes, measuring frontier position as calibration data shrinks. That
specific experiment (Item 2) appears to be the genuinely novel contribution
available here.

## What this means for the rest of the task

1. **The framing must change.** "A new conditional conformal method" is not
   defensible. "A validated low-dimensional supervised geometry that reaches a
   better coverage-width frontier position with far less calibration data" is
   defensible — and it is exactly what Item 2 measures.
2. **RLCP should be in the comparison.** Item 5's baseline set is incomplete
   without at least one localized/kernel-weighted conformal method. It is not
   currently implemented in this project.
3. **Item 3's expected value drops.** If localized quantile regression already
   does what `q_α(z)` would do, building it re-derives prior work. Its value now
   rests on whether it closes the coverage gap that the fixed multiplier could
   not — a narrower, purely empirical question.

## Sources

- [Conformal prediction with local weights: randomization enables robust guarantees (RLCP)](https://arxiv.org/html/2310.07850v2)
- [Localized Conformal Prediction: A Generalized Inference Framework (Biometrika)](https://academic.oup.com/biomet/article/110/1/33/6647831)
- [SpeedCP: Fast Kernel-based Conditional Conformal Prediction](https://arxiv.org/pdf/2509.24100)
- [Spatial Conformal Inference through Localized Quantile Regression](https://arxiv.org/html/2412.01098v1)
- [Beyond Marginal Validity: Finite-Sample Guarantees for Localized Conformal Prediction](https://arxiv.org/html/2608.06206)
- [Enhanced localized conformal prediction with imperfect auxiliary information](https://arxiv.org/pdf/2606.08551)
- [Localized Conformal Prediction (overview)](https://www.emergentmind.com/topics/localized-conformal-prediction)
- [Probabilistic Conformal Coverage Guarantees in Small-Data Settings](https://arxiv.org/html/2509.15349v1)
