import numpy as np, matplotlib.pyplot as plt
# cosines from stage_mech (seed 0), layer 21 and last layer
rows = [("base model (no training)","base",0.45,0.48),
        ("chat refusal SFT","chat SFT",0.54,0.35),
        ("chat reasoning-first SFT (reason_first_v2)","chat SFT",0.55,0.31),
        ("+ benign tool calls only (rfx_benign)","crossed SFT",0.61,0.38),
        ("+ crossed harm x format (rfx_factorial)","crossed SFT",0.65,0.45),
        ("+ crossed, plain replies (rfx_plain_factorial)","crossed SFT",0.69,0.37),
        ("+ crossed, mixed refusal formats (rfx_mixfmt)","crossed SFT",0.61,0.42),
        ("+ crossed + supervised cross-view (rfx_factorial_sc)","crossed SFT",0.74,0.78),
        ("+ RL, format-agnostic reward (rl_decide_factorial)","RL",0.65,0.44)]
ang = lambda c: np.degrees(np.arccos(c))
fig, ax = plt.subplots(figsize=(9.2,4.8), dpi=200)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
y = np.arange(len(rows))[::-1]
for yi,(lab,g,c21,cl) in zip(y,rows):
    ax.plot([ang(cl),ang(c21)],[yi,yi],color="#c9c8c2",lw=2,zorder=1)
ax.scatter([ang(r[3]) for r in rows],y,s=60,color="#2a78d6",zorder=3,label="last layer (36)",edgecolor="#fcfcfb",linewidth=2)
ax.scatter([ang(r[2]) for r in rows],y,s=60,color="#eb6834",zorder=3,label="layer 21",edgecolor="#fcfcfb",linewidth=2)
ax.axvline(90,color="#52514e",lw=1,ls="--"); ax.text(89.2,y[0]+0.45,"90° = orthogonal\n(unrelated)",ha="right",va="top",fontsize=7.5,color="#52514e")
ax.axvline(0,color="#52514e",lw=0.6)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=8,color="#0b0b0b")
ax.set_xlim(0,92); ax.set_xlabel("angle between chat and agent harmful−harmless directions (degrees)  ← more aligned",fontsize=8.5,color="#52514e")
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="x",labelsize=8,colors="#52514e"); ax.tick_params(axis="y",length=0)
ax.grid(axis="x",color="#e6e5e0",lw=0.6); ax.set_axisbelow(True)
r=rows[7]; ax.annotate("39°: only supervised cross-view\nbrings the late-layer directions together",(ang(r[3]),y[7]),xytext=(8,y[7]+1.6),fontsize=7.5,color="#0b0b0b",arrowprops=dict(arrowstyle="-",color="#8a8984",lw=0.8))
ax.set_title("How aligned are chat and agent harm directions? (Tiny Aya Base, held-out harm categories, seed 0)",fontsize=9.5,loc="left",color="#0b0b0b")
ax.legend(fontsize=8,frameon=False,loc="lower left",bbox_to_anchor=(0,-0.28),ncol=2)
plt.tight_layout(); plt.savefig("fig_direction_angle_by_stage.png",facecolor=fig.get_facecolor())
