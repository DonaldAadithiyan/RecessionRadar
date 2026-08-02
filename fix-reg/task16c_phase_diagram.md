# Task 16C — Synthetic Phase Diagram

**Verdict: the phase diagram was produced, and it does NOT validate the
synthetic section — it undermines it. At the coordinates the real domains
actually occupy, the synthetic model predicts rare-count should beat diversity.
Every real domain measures the opposite. This is a finding about the synthetic
construction's fidelity, not about the real mechanism, and the paper should
treat §5.2 accordingly.**

Script: `fix-reg/task16c_phase_diagram.py`.
Data: `task16c_phase_diagram.csv`, `task16c_real_domain_coords.csv`.
Figure: `figures/task16c_phase_diagram.{pdf,png}`.

---

## Two construction problems found and fixed before reporting

**1. The spec's suggested Δ_S grid saturates.** At {0, 1, 2, 4, 8, 16}, the
columns 4/8/16 returned *numerically identical* gaps at every frequency.
Diagnosed directly: beyond Δ_S ≈ 2, **100% of rare scores already exceed the
pool's 90th percentile**, so pushing them further apart changes nothing ACI
sees. Those cells carry no information.

The grid was refined to {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3, 4, 8} —
dense where the variation and the real domains live, retaining 4 and 8 to *show*
the plateau rather than hide it. 66 cells, 200 draws each.

**2. Δ_S needed defining.** The spec asks for mean separation in units of the
normal-score SD; Phase 4a parameterised by a scale multiplier instead. Δ_S as
specified is used here, because it is interpretable and lets the real domains be
placed on the same axes.

## The grid as measured

ρ(diversity) − ρ(rare-count); positive means diversity dominates. Reported as
measured, with no fitted boundary, per the guardrail.

| freq \ Δ_S | 0 | 0.5 | 1.0 | 1.25 | 1.5 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|---|---|
| 2% | +0.53 | +0.50 | +0.47 | +0.30 | +0.28 | +0.20 | +0.23 | +0.23 |
| 5% | +0.45 | +0.47 | +0.38 | +0.26 | +0.18 | −0.05 | −0.01 | +0.02 |
| 8% | +0.38 | +0.47 | +0.14 | −0.02 | +0.00 | −0.16 | −0.14 | −0.14 |
| 12% | +0.51 | +0.39 | +0.07 | +0.02 | −0.03 | −0.24 | −0.36 | −0.35 |
| 20% | +0.41 | +0.32 | +0.21 | +0.06 | +0.02 | −0.07 | −0.07 | −0.07 |
| 30% | +0.52 | +0.32 | +0.28 | +0.13 | +0.18 | +0.21 | +0.15 | +0.15 |

The refined grid shows a clean structure the coarse one hid: **a monotone decay
in Δ_S, crossing zero around Δ_S ≈ 1.25–2.0** at mid frequencies. 50 of 66
non-degenerate cells are positive.

## Where the real domains sit — and why that is a problem

Measured from actual out-of-fold scores:

| Domain | Frequency | Δ_S | Synthetic prediction | **Real measured gap** |
|---|---|---|---|---|
| Recession 6M | 8.4% | **1.26** | ≈ −0.02 | **+0.442** |
| Recession 3M | 8.4% | **2.19** | ≈ −0.16 | **+0.402** |
| Healthcare | 10.3% | **2.41** | ≈ −0.24 | **+0.112 / +0.132** |
| Climate | 10.7% | **2.97** | ≈ −0.36 | **+0.005 / +0.118** |

**All four real domains land in the negative region of the synthetic map, and
all four measure positive gaps in reality.** The synthetic model gets the sign
wrong at every real coordinate.

### What this does and does not mean

**It does NOT weaken the empirical finding.** The real-domain results (Tasks 1,
7, 8, 13, 16A) stand on their own measurements. Diversity dominates rare-count
in 12 of 12 temporal cells and 7 of 8 model cells; nothing in a simulation
overturns a measurement.

**It DOES mean the synthetic section cannot be used as corroboration.** §5.2
currently presents seven synthetic scenarios as evidence that the mechanism
generalises. Mapped properly, the same construction *contradicts* the real
results at the real parameters. Presenting it as supporting evidence would be
selective reading of a model that fails its own validation check.

**The most likely cause is the synthetic construction, not the mechanism.** The
generator draws rare and normal scores as independent half-normals with a mean
shift. Real nonconformity scores are neither independent (they are serially
correlated and regime-clustered) nor two-component (they are a continuum). In
particular, a clean two-component mixture makes rare-count an unusually *good*
predictor by construction — knowing how many rare draws you took tells you
almost exactly what the upper tail looks like — which is precisely the advantage
rare-count enjoys in the blue region and does not enjoy in real data.

## Recommendation for the paper

1. **Do not cite the synthetic grid as support for the main finding.** Report it
   as a validation check the synthetic model *failed*, and say what that implies:
   the two-component simulation is too simple to reproduce the real mechanism.
2. **Reframe §5.2.** Its seven scenarios are not wrong, but they sample a region
   where the construction happens to agree; the full map shows that agreement
   does not extend to the real domains' coordinates.
3. **Keep the figure.** It is honest and informative — it shows exactly where the
   simulation and reality part company, which is more useful than seven points
   that concealed it.

## Honest caveats

- **Exploratory, as specified.** No parametric boundary was fitted; the grid is
  reported as measured.
- **One seed per cell.** Cell-to-cell noise is visible (e.g. the 30% row is
  non-monotone), and some sign flips near zero could reverse under a different
  seed. The overall Δ_S decay is robust to that; individual near-zero cells are
  not.
- **The real-domain coordinates are point estimates.** Δ_S is computed from one
  set of out-of-fold scores per domain; no interval is placed on it, and Δ_S
  varies with horizon (recession 3M = 2.19 vs 6M = 1.26) more than across
  domains.
- **The mismatch is diagnosed, not proven.** The explanation above (independence
  and two-component structure) is the most plausible cause given the
  construction, but no experiment here isolates which assumption is responsible.
  A generator with correlated, continuum-valued scores would test it.
