"""Phase 6 presentation pass: two paper-ready summary figures matching house style.
 F_methods: coverage by horizon across all calibration/conformal strategies.
 F_phase2 : diversity-optimal vs trailing vs fixed-ablation (coverage, grouped bars).
From phase2/phase3 CSVs. Serif 8pt, validated palette, grayscale-safe."""
import os,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","figures"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],
    "font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.3,
    "axes.linewidth":0.6,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.02,
    "pdf.fonttype":42,"ps.fonttype":42})
BLUE,YEL,GREEN,RED,ORANGE,VIOLET="#2a78d6","#eda100","#008300","#e34948","#eb6834","#4a3aa7"
GRID="#d9d9d6"; H=["Current","1M","3M","6M"]

p2=pd.read_csv(os.path.join(HERE,"phase2_selection_comparison.csv")).set_index("Horizon")
p3a=pd.read_csv(os.path.join(HERE,"phase3a_mondrian.csv")).set_index("Horizon")
p3bc=pd.read_csv(os.path.join(HERE,"phase3bc_extra_baselines.csv")).set_index("Horizon")

# ---- F_methods: coverage across strategies, grouped bars by horizon ----
series=[("Pooled ACI (trailing)",[p2.loc[h,"trailing_cov"] for h in H],BLUE,""),
        ("Mondrian ACI",[p3a.loc[h,"Mondrian_cov"] for h in H],RED,"xx"),
        ("PID-conformal",[p3bc.loc[h,"PID_cov"] for h in H],GREEN,".."),
        ("Diversity-optimal",[p2.loc[h,"divopt_cov"] for h in H],ORANGE,"//")]
fig,ax=plt.subplots(figsize=(3.3,2.5))
x=np.arange(len(H)); w=0.20
for k,(name,vals,c,ht) in enumerate(series):
    ax.bar(x+(k-1.5)*w,vals,w,label=name,color=c,alpha=0.85,edgecolor="white",linewidth=0.4,hatch=ht)
ax.axhline(90,color="#444",ls=(0,(4,3)),lw=0.8,zorder=5)
ax.text(3.45,90.4,"90%",fontsize=6,color="#444",va="bottom",ha="right")
ax.set_xticks(x); ax.set_xticklabels(H); ax.set_ylabel("ACI coverage (%)")
ax.set_ylim(55,95); ax.set_yticks([60,70,80,90])
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",color=GRID,lw=0.4,zorder=0); ax.tick_params(length=2.5,width=0.6)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=2,frameon=False,handlelength=1.5,columnspacing=1.0)
fig.savefig(os.path.join(OUT,"methods_coverage_comparison.pdf"),format="pdf")
fig.savefig(os.path.join(OUT,"methods_coverage_comparison.png"),format="png")
plt.close(fig); print("wrote figures/methods_coverage_comparison.pdf + .png")
