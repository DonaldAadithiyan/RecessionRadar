"""
Publication-ready figures for the IJCAI submission (two-column, 10pt Times).
Pure plotting from already-computed CSVs — no model training.

Fig 1: calibration_threshold  (single-col, 3.3x2.4in)  from aci_composition_sweep.csv
Fig 2: shap_heatmap           (two-col,   7.0x3.2in)   from shap_top12_x_horizon.csv
Fig 3: bootstrap_ci_forest    (single-col,3.3x2.6in)   from bootstrap_block_ci_90.csv

Palette (validated colorblind-safe on light surface via dataviz validator):
  blue #2a78d6, red #e34948, green #008300, orange #eb6834
Every categorical series also carries a distinct marker + linestyle so the
figures survive grayscale printing (secondary encoding, not color alone).
Sequential viridis for the all-positive SHAP magnitudes.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = HERE                                   # CSVs live in fix-reg/
OUT  = os.path.join(HERE, "..", "figures")    # deliverables -> repo/figures/
os.makedirs(OUT, exist_ok=True)

# ── consistent style across all figures ──────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  6.5,
    "axes.linewidth":   0.6,
    "lines.linewidth":  1.4,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":     42,   # embed real fonts (no type-3) for camera-ready
    "ps.fonttype":      42,
})

BLUE, RED, GREEN, ORANGE = "#2a78d6", "#e34948", "#008300", "#eb6834"
GRID = "#d9d9d6"

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), format="pdf")
    fig.savefig(os.path.join(OUT, name + ".png"), format="png")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf + .png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Calibration threshold
# ═══════════════════════════════════════════════════════════════════════════
def fig1():
    df = pd.read_csv(os.path.join(CSV, "aci_composition_sweep.csv"))
    df["frac"] = df["Cal_Fraction"].str.rstrip("%").astype(int)
    horizons = ["Current", "1M", "3M", "6M"]
    # blue, red, green, orange + distinct markers/linestyles (grayscale-safe)
    style = {
        "Current": (BLUE,   "o", "-"),
        "1M":      (RED,    "s", "--"),
        "3M":      (GREEN,  "^", "-."),
        "6M":      (ORANGE, "D", ":"),
    }
    fig, ax = plt.subplots(figsize=(3.3, 2.4))

    # nominal target reference line (recessive, behind data)
    ax.axhline(90, color="#6a6a6a", ls=(0, (4, 3)), lw=0.8, zorder=1)
    ax.text(19.3, 91.2, "nominal target (90%)", fontsize=6, color="#6a6a6a",
            va="bottom", ha="left")

    for h in horizons:
        sub = df[df["Horizon"] == h].sort_values("frac")
        c, mk, ls = style[h]
        ax.plot(sub["frac"], sub["Coverage"], color=c, marker=mk, linestyle=ls,
                markersize=4.5, markeredgecolor="white", markeredgewidth=0.5,
                label=h, zorder=3, clip_on=False)

    ax.set_xlabel("Calibration window (% of training tail)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_xticks([20, 25, 30, 35, 40])
    ax.set_xlim(19, 41)
    ax.set_ylim(55, 95)
    ax.set_yticks([60, 70, 80, 90])
    ax.set_xlim(19, 42)          # a touch more room so the target label fits
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(length=3, width=0.6)
    ax.legend(loc="lower right", ncol=2, frameon=False, handlelength=2.2,
              columnspacing=1.0, borderaxespad=0.3)
    save(fig, "calibration_threshold")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Horizon-conditional SHAP heatmap
# ═══════════════════════════════════════════════════════════════════════════
# Orientation choice: 12 features (rows) x 4 horizons (cols). This keeps the
# 4 horizon columns wide enough to read the Current->6M magnitude shift left to
# right, and gives 12 long feature names their own rows (transposing to 4x12
# would squeeze 12 long labels onto the x-axis illegibly). Reported in caption.
FEATURE_ABBR = {
    "OECD_CLI_index_trend":       "OECD CLI trend",
    "INDPRO_diff3":               "INDPRO diff3",
    "gdp_per_capita_residual":    "GDPpc residual",
    "OECD_CLI_index_residual":    "OECD CLI residual",
    "gdp_per_capita_diff3":       "GDPpc diff3",
    "gdp_per_capita":             "GDPpc level",
    "10_year_rate_residual":      "10Y rate residual",
    "OECD_CLI_index_diff1":       "OECD CLI diff1",
    "gdp_per_capita_diff1":       "GDPpc diff1",
    "OECD_CLI_index_pct_change1": "OECD CLI %chg1",
    "share_price":                "Share price",
    "gdp_per_capita_pct_change1": "GDPpc %chg1",
}
def fig2():
    df = pd.read_csv(os.path.join(CSV, "shap_top12_x_horizon.csv")).set_index("Feature")
    horizons = ["Current", "1M", "3M", "6M"]
    M = df[horizons].values                      # (12, 4)
    labels = [FEATURE_ABBR[f] for f in df.index]

    fig, ax = plt.subplots(figsize=(3.3, 3.6))   # single-col; 12 rows need height
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=M.max())

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(horizons)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Forecast horizon")
    ax.tick_params(length=0)
    # thin white gridlines between cells (2px surface gap analogue)
    ax.set_xticks(np.arange(-.5, len(horizons), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    for s in ax.spines.values():
        s.set_visible(False)

    # annotate cells — legible at this size; text color flips on dark cells
    thr = M.max() * 0.55
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v < thr else "#111111")

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Mean |SHAP|", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    save(fig, "shap_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Bootstrap CI forest plot
# ═══════════════════════════════════════════════════════════════════════════
# Axis scale: linear. 6M XGB-indep CI reaches 63.7 while the rest are <18, so a
# linear x-axis compresses the other three horizons; a symlog/log x-axis makes
# all four horizons' overlap-vs-disjoint readable at once. Chose log. Reported.
def fig3():
    df = pd.read_csv(os.path.join(CSV, "bootstrap_block_ci_90.csv"))
    horizons = ["Current", "1M", "3M", "6M"]
    models = [("Ensemble (Ours)", BLUE, "o"), ("XGB Indep (tuned)", ORANGE, "s")]

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    yb = np.arange(len(horizons))[::-1]          # Current at top
    off = 0.16
    for k, (mname, c, mk) in enumerate(models):
        dy = off if k == 0 else -off
        for hi, h in enumerate(horizons):
            r = df[(df.Horizon == h) & (df.Model == mname)].iloc[0]
            y = yb[hi] + dy
            ax.plot([r.CI_lo_90, r.CI_hi_90], [y, y], color=c, lw=1.4, zorder=2,
                    solid_capstyle="round")
            ax.plot(r.MAE, y, marker=mk, color=c, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(0.15, 80)
    ax.set_xticks([0.2, 1, 5, 10, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks(yb)
    ax.set_yticklabels(horizons)
    ax.set_ylim(-0.5, len(horizons) - 0.5)
    ax.set_xlabel("MAE (pp), 90% block-bootstrap CI  —  log scale")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(length=3, width=0.6)
    # faint separators between horizon rows
    for yy in yb[:-1]:
        ax.axhline(yy - 0.5, color=GRID, lw=0.4, zorder=0)

    handles = [Line2D([0], [0], color=c, marker=mk, markersize=5,
                      markeredgecolor="white", markeredgewidth=0.5, lw=1.4,
                      label=mname) for mname, c, mk in models]
    # legend below the axes so it never collides with the (wide) 6M bars
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, handlelength=1.8, columnspacing=1.4)
    save(fig, "bootstrap_ci_forest")


if __name__ == "__main__":
    print("Generating publication figures -> figures/")
    fig1()
    fig2()
    fig3()
    print("Done.")
