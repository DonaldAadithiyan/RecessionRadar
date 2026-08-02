"""
TASK 16D figure — the diversity advantage across target coverage levels.

Two panels:
  left  — rho(support width, coverage) and rho(rare count, coverage) vs alpha
  right — the gap between them, with the region ACI actually operates in
          ([0.073, 0.132], measured in Task 10) shaded, so a reader can see
          immediately that the paper's evidence lives inside the shaded band and
          the decay happens outside it.

Output: figures/task16d_alpha_sweep.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

D = pd.read_csv("fix-reg/task16d_alpha_sweep.csv")
HORIZONS = ["1M", "3M", "6M", "Current"]
COLORS = {"Current": "#9aa7b4", "1M": "#5a7fa8", "3M": "#c07830", "6M": "#1a1a1a"}

# ACI's measured operating range (Task 10)
ACI_LO, ACI_HI = 0.073, 0.132

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))

# ── Left: the two correlations ─────────────────────────────────────────────
ax = axes[0]
for h in HORIZONS:
    d = D[D["horizon"] == h].sort_values("alpha")
    if d.empty:
        continue
    ax.plot(d["alpha"], d["rho_supp_cov"], "-o", ms=5, color=COLORS[h],
            label=f"{h} — support width")
    ax.plot(d["alpha"], d["rho_rare_cov"], "--s", ms=4, color=COLORS[h],
            alpha=.55, label=f"{h} — rare count")
ax.axvspan(ACI_LO, ACI_HI, color="#f0c040", alpha=.22, zorder=0)
ax.axhline(0, color="#888", lw=.9, ls=":")
ax.set_xlabel(r"target miscoverage $\alpha$", fontsize=10.5)
ax.set_ylabel(r"$\rho$ with realised coverage", fontsize=10.5)
ax.set_title("Both predictors weaken as the target loosens", fontsize=11.5)
ax.grid(alpha=.25, lw=.7)
ax.set_axisbelow(True)
ax.legend(fontsize=7.2, ncol=2, loc="lower left", framealpha=.9)

# ── Right: the gap ─────────────────────────────────────────────────────────
ax = axes[1]
for h in HORIZONS:
    d = D[D["horizon"] == h].sort_values("alpha")
    if d.empty:
        continue
    ax.plot(d["alpha"], d["gap"], "-o", ms=6, color=COLORS[h], label=h,
            lw=2.0 if h == "6M" else 1.4)
ax.axvspan(ACI_LO, ACI_HI, color="#f0c040", alpha=.22, zorder=0,
           label="ACI's measured range")
ax.axhline(0, color="crimson", lw=1.3, ls="--")
ax.set_xlabel(r"target miscoverage $\alpha$", fontsize=10.5)
ax.set_ylabel(r"$\rho$(diversity) $-$ $\rho$(rare-count)", fontsize=10.5)
ax.set_title("The diversity advantage decays as $\\alpha$ rises\n"
             "(above the red line = diversity dominates)", fontsize=11.5)
ax.grid(alpha=.25, lw=.7)
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, framealpha=.9)

sec = ax.secondary_xaxis("top", functions=(lambda a: 100 * (1 - a),
                                           lambda c: 1 - c / 100))
sec.set_xlabel("nominal coverage %", fontsize=9.5)

fig.suptitle("Does the calibration-diversity finding hold at other target "
             "coverage levels?  (recession testbed, 200 draws per point)",
             fontsize=12.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/task16d_alpha_sweep.{ext}", dpi=170, bbox_inches="tight")
print(f"Saved {OUT}/task16d_alpha_sweep.pdf / .png")
