"""P1 — nonconformity-score distribution by regime (expansion vs rare-event).
Boxplots, 4 horizon facets, log y (scores span ~0.01 to ~26 pp), p90 marked.
Matches paper style: serif 8pt, blue(#2a78d6)/orange(#eb6834), grayscale-safe
via hatching. From fix-reg/nonconformity_scores_raw.csv — no model training."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","figures")
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],
    "font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.5,
    "axes.linewidth":0.6,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight",
    "savefig.pad_inches":0.02,"pdf.fonttype":42,"ps.fonttype":42})
BLUE,ORANGE,GRID="#2a78d6","#eb6834","#d9d9d6"

df=pd.read_csv(os.path.join(HERE,"nonconformity_scores_raw.csv"))
horizons=["Current","1M","3M","6M"]

fig,axes=plt.subplots(1,4,figsize=(7.0,2.3),sharey=True)
for ax,h in zip(axes,horizons):
    sub=df[df.Horizon==h]
    exp=sub[sub.regime=="expansion"]["score"].values
    rare=sub[sub.regime=="rare"]["score"].values
    data=[exp,rare]; cols=[BLUE,ORANGE]; hatches=["","///"]
    bp=ax.boxplot(data,positions=[1,2],widths=0.6,patch_artist=True,showfliers=True,
                  flierprops=dict(marker="o",markersize=1.5,markerfacecolor="#888",
                                  markeredgecolor="none",alpha=0.5),
                  medianprops=dict(color="#111",lw=1.0),
                  whiskerprops=dict(color="#555",lw=0.7),capprops=dict(color="#555",lw=0.7))
    for patch,c,ht in zip(bp["boxes"],cols,hatches):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(c); patch.set_linewidth(0.8)
        if ht: patch.set_hatch(ht)
    # p90 markers (the quantile ACI uses)
    for pos,arr,c in zip([1,2],data,cols):
        p90=np.percentile(arr,90)
        ax.plot([pos-0.32,pos+0.32],[p90,p90],color=c,lw=1.3,ls=(0,(2,1.5)),zorder=5)
    ax.set_yscale("log")
    ax.set_title(h,fontsize=8)
    ax.set_xticks([1,2]); ax.set_xticklabels(["Exp.","Rare"])
    ax.grid(axis="y",color=GRID,lw=0.4,which="both")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5,width=0.6)
axes[0].set_ylabel("Nonconformity score\n|actual − pred| (pp, log)")
handles=[Patch(facecolor=BLUE,alpha=0.55,edgecolor=BLUE,label="Expansion (<50%)"),
         Patch(facecolor=ORANGE,alpha=0.55,edgecolor=ORANGE,hatch="///",label="Rare-event (≥50%)"),
         plt.Line2D([0],[0],color="#555",lw=1.3,ls=(0,(2,1.5)),label="90th pctile (ACI width)")]
fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.06),ncol=3,
           frameon=False,handlelength=1.8,columnspacing=1.3)
fig.savefig(os.path.join(OUT,"nonconformity_by_regime.pdf"),format="pdf")
fig.savefig(os.path.join(OUT,"nonconformity_by_regime.png"),format="png")
plt.close(fig)
# export the exact CSV used
df.to_csv(os.path.join(HERE,"nonconformity_scores_raw.csv"),index=False)
print("wrote figures/nonconformity_by_regime.pdf + .png")
