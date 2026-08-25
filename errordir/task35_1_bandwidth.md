# Task 35, Item 1 — Bandwidth Rule, Fixed in Advance

**Chosen and stated before fitting D(x) on any domain. Applied identically to
every domain and model class in this task. No domain-specific tuning, no
adjustment after seeing results.**

## What went wrong in Task 33

Task 33 used the **median pairwise distance** among FIT points as the Gaussian
kernel bandwidth — a standard, defensible, fit-period-only heuristic. It
produced `h = 4.21–4.36`.

The problem is dimensional. In `d` standardized dimensions the median pairwise
distance is ≈ √(2d), which for d = 12–21 is **4.9–6.5**. So the chosen `h` was
the same order as the typical inter-point distance, and a Gaussian kernel at that
width averages over most of the sample. D(x) became a near-global mean-error
surface and ∇D(x) barely varied — only **3–10%** of tree points showed meaningful
gradient variation, and the ridge control failed the same way.

The heuristic was not wrong as a heuristic; it was wrong for *this* purpose. The
median distance is designed to make a kernel see a representative slice of the
data, which is the opposite of what a locality test needs.

## The rule chosen for this task

**Dimension-adjusted Silverman's rule**, one of the two options Task 33's own
report named as the honest follow-up:

    h = sigma_eff * ( 4 / (d + 2) )^(1/(d+4)) * n^(-1/(d+4))

where `sigma_eff` is the mean per-dimension standard deviation of the FIT
projections (1.0 by construction after standardization, computed rather than
assumed), `d` is the feature dimension, and `n` is the number of FIT points used
in the kernel.

**Why this rule rather than LOO likelihood cross-validation:**

1. **It is closed-form and parameter-free.** LOO-CV requires choosing a search
   grid and an objective, each of which is a further decision that would itself
   need justifying without test data. Silverman introduces no such choice.
2. **It is explicitly dimension-aware**, which is precisely the failure mode from
   Task 33. The `n^(-1/(d+4))` term shrinks the bandwidth as the sample grows,
   and the `(4/(d+2))^(1/(d+4))` factor corrects for dimension directly.
3. **It is the standard default** for kernel density and Nadaraya-Watson
   estimation, so it is a principled prior choice rather than one reverse-
   engineered to produce locality.

**Honest caveat, stated now:** Silverman's rule is derived for density estimation
under a Gaussian reference distribution, and it is known to over-smooth in high
dimensions. It may well still produce too large a bandwidth at d = 21. That is
exactly what **Item 2's locality check exists to detect**, and if it fails there
the honest outcome is Item 4's outcome 3 — not a third bandwidth attempt within
this task.

## Locality criterion, also fixed in advance

Item 2 gates on whether ∇D(x) genuinely varies across points. Task 33's failure
signature was gradients that were nearly identical everywhere. The criterion:

    mean pairwise cosine similarity between per-point unit gradients

    LOCAL      if mean cosine similarity < 0.90
    NOT LOCAL  if >= 0.90   (near-constant direction field)

The 0.90 threshold is set from Task 33's observed failure: gradients there were
effectively collinear, and a field where the average pair of gradients points
within ~26 degrees of each other cannot supply meaningfully different per-point
directions. A secondary diagnostic — the fraction of points whose gradient is
more than 30 degrees from the mean gradient — is reported alongside for
interpretability, but the pass/fail is on the cosine criterion alone.

## Application scope

Applied identically to: insurance and energy, × {ridge, xgboost, lightgbm}.
Ridge is the control arm on every domain, per this project's discipline since
Task 25 — and per Item 2's guardrail, **a ridge failure blocks the tree tests
entirely**.
