"""
TASK 11 (step 1) — Are the nonconformity-score tails actually extrapolatable?

Pool augmentation only makes sense if a fitted tail can reach BEYOND the
observed maximum. That is an empirical property of each domain's score
distribution, and it is checked here before any augmentation is built.

For each domain/model/horizon this reports:
  - the GPD shape parameter xi fitted above several thresholds
  - whether xi > 0 (heavy tail, extrapolates upward without bound),
    xi ~ 0 (exponential tail, extrapolates slowly),
    or xi < 0 (BOUNDED tail — the fitted distribution has a finite endpoint,
    so augmentation cannot exceed it and the idea fails for that cell)
  - the implied endpoint when xi < 0, versus the observed max
  - a stability check across thresholds (a well-specified GPD has roughly
    constant xi as the threshold rises; wild variation means the fit is noise)

If xi < 0 with an endpoint near the observed max, augmentation is pointless for
that cell and this is reported plainly rather than pushed through.

Output: fix-reg/task11_tail_diagnostics.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()

OUT = "fix-reg"
N_FIX = 254
THRESHOLDS = [0.75, 0.80, 0.85, 0.90]

rows = []


def diagnose(domain, model, horizon, scores):
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 60:
        print(f"  [skip] {domain}/{model}/{horizon}: only {len(s)} scores")
        return
    obs_max = float(s.max())

    fits = []
    for q in THRESHOLDS:
        u = float(np.quantile(s, q))
        exc = s[s > u] - u
        if len(exc) < 15:
            continue
        try:
            xi, loc, sigma = stats.genpareto.fit(exc, floc=0)
        except Exception:
            continue
        # Finite upper endpoint exists only when xi < 0: u - sigma/xi
        endpoint = (u - sigma / xi) if xi < -1e-9 else np.inf
        fits.append(dict(threshold_q=q, u=round(u, 4), n_exc=int(len(exc)),
                         xi=round(float(xi), 4), sigma=round(float(sigma), 4),
                         endpoint=(round(float(endpoint), 4)
                                   if np.isfinite(endpoint) else None)))

    if not fits:
        print(f"  [skip] {domain}/{model}/{horizon}: no fittable threshold")
        return

    xis = [f["xi"] for f in fits]
    xi_spread = max(xis) - min(xis)
    # Use the 80th-percentile fit as the headline (Phase 3b's convention)
    head = next((f for f in fits if f["threshold_q"] == 0.80), fits[0])

    if head["xi"] > 0.05:
        verdict = "HEAVY (extrapolates upward)"
    elif head["xi"] > -0.05:
        verdict = "EXPONENTIAL-ish (extrapolates slowly)"
    else:
        verdict = "BOUNDED (finite endpoint)"

    headroom = None
    if head["endpoint"] is not None:
        headroom = round(head["endpoint"] / obs_max, 3)

    rows.append(dict(domain=domain, model=model, horizon=horizon,
                     n_scores=int(len(s)), obs_max=round(obs_max, 4),
                     xi=head["xi"], sigma=head["sigma"], u=head["u"],
                     n_exceedances=head["n_exc"],
                     endpoint=head["endpoint"],
                     endpoint_over_obsmax=headroom,
                     xi_spread_across_thresholds=round(float(xi_spread), 4),
                     verdict=verdict))
    r = rows[-1]
    ep = ("inf" if r["endpoint"] is None else f"{r['endpoint']:.2f}")
    print(f"  {domain:11s} {model:14s} {horizon:12s} "
          f"xi={r['xi']:+.4f}  endpoint={ep:>9s}  "
          f"obs_max={r['obs_max']:8.2f}  xi-spread={r['xi_spread_across_thresholds']:.3f}"
          f"  -> {verdict}")


print("=" * 96)
print("TASK 11 step 1 — GPD tail diagnostics on the nonconformity scores")
print("=" * 96)
print("  xi > 0  : heavy tail, synthetic scores can exceed the observed max")
print("  xi ~ 0  : exponential tail, slow upward extrapolation")
print("  xi < 0  : BOUNDED — fitted tail has a finite endpoint; augmentation")
print("            cannot reach past it, so the idea fails for that cell")
print()

# ── Recession ───────────────────────────────────────────────────────────────
import task_oof_and_probit as T  # noqa: E402
print("Recession (out-of-fold scores):")
for h_idx, h in enumerate(["Current", "1M", "3M", "6M"]):
    diagnose("Recession", "stacking-chain", h,
             np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx]))

# ── Healthcare ──────────────────────────────────────────────────────────────
import runpy  # noqa: E402
print("\nHealthcare:")
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    diagnose("Healthcare", name, "30-day", g["SCORES"][name])

# ── Climate ────────────────────────────────────────────────────────────────
print("\nClimate:")
gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    diagnose("Climate", name, "region-month", gc["SCORES"][name])

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/task11_tail_diagnostics.csv", index=False)

print("\n" + "=" * 96)
print("VERDICT — is pool augmentation viable?")
print("=" * 96)
print(df[["domain", "model", "horizon", "xi", "endpoint",
          "endpoint_over_obsmax", "xi_spread_across_thresholds",
          "verdict"]].to_string(index=False))

n_bounded = int((df["verdict"].str.startswith("BOUNDED")).sum())
n_unstable = int((df["xi_spread_across_thresholds"] > 0.5).sum())
print(f"\n  Cells with a BOUNDED fitted tail: {n_bounded} of {len(df)}")
print(f"  Cells with unstable xi across thresholds (>0.5): {n_unstable} of {len(df)}")
if n_bounded == len(df):
    print("\n  ALL cells bounded -> augmentation cannot extend any tail.")
    print("  The idea would be dead on arrival; report and stop.")
elif n_bounded:
    print("\n  Mixed: augmentation is only meaningful where the tail is not")
    print("  bounded below the observed max. Cells will be handled accordingly.")
print(f"\nSaved {OUT}/task11_tail_diagnostics.csv")
