"""arch_grtx.png — grtx 架构(shared encoder + context encoder → cost volume → RAFT)。
hw 居中, 同喂 shared encoder(与tw)和 context encoder; context enc 产出 h/inp 喂 RAFT 的 ConvGRU。
自包含 schematic。用法: python3 gen_arch_grtx.py"""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _common import OUT
plt.rcParams.update({'font.size':9.5,'font.family':'DejaVu Sans'})
W,H=15.4,15.8
fig,ax=plt.subplots(figsize=(13.6,14.0)); ax.set_xlim(0,W); ax.set_ylim(-0.7,H); ax.axis('off')
COL=dict(inp='#dCEBFb',trunk='#bcd6f5',tx='#c9d2f7',cost='#a9c4ef',read='#fde39a',
         raftbg='#eafaf0',raft='#bdeccb',ctx='#fde1b0',out='#bbf7d0',mono='#0f172a')
def box(x,y,w,h,title,shape=None,fc='#eee',fs=9.3,tfs=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",fc=fc,ec='#1e293b',lw=1.3,zorder=3))
    if shape:
        ax.text(x+w/2,y+h*0.64,title,ha='center',va='center',fontsize=fs,zorder=4,weight='bold')
        ax.text(x+w/2,y+h*0.27,shape,ha='center',va='center',fontsize=(tfs or fs-0.7),zorder=4,family='monospace',color='#b91c1c')
    else:
        ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=fs,zorder=4,weight='bold')
def arr(x1,y1,x2,y2,rad=0,c='#1e293b',lw=1.7,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=15,lw=lw,color=c,connectionstyle=f"arc3,rad={rad}",zorder=5,linestyle=ls))

# ── 输入行: tw(左) · hw(中) · context encoder(右; 输入=hw)──
box(0.4,14.6,3.0,1.0,"Typewell (tw)","[B, 2, 320]  GR + 0·ΔZ pad",COL['inp'],9.0)
box(4.0,14.6,3.0,1.0,"Horizontal well (hw)","[B, 2, 768]  GR+ΔZ",COL['inp'],9.0)
# context encoder(第二个 encoder)
ax.add_patch(FancyBboxPatch((8.4,13.55),4.6,1.95,boxstyle="round,pad=0.02,rounding_size=0.05",fc=COL['ctx'],ec='#1e293b',lw=1.3,zorder=3))
cxt=10.7
ax.text(cxt,15.12,"Context encoder",ha='center',fontsize=9.5,weight='bold',zorder=4)
ax.text(cxt,14.60,"in: [GR, ΔZ, is_known, known/dmax] [B,4,768]",ha='center',fontsize=8.0,family='monospace',color='#7c2d12',zorder=4)
ax.text(cxt,14.22,"pool÷8 → conv →",ha='center',fontsize=8.0,family='monospace',color='#7c2d12',zorder=4)
ax.text(cxt,13.84,"h [B,H,96] (tanh)  ‖  inp [B,H,96] (relu)",ha='center',fontsize=8.0,family='monospace',color='#7c2d12',zorder=4)

# ── shared encoder 主干(左竖流)──
box(0.7,13.15,6.5,1.0,"Shared conv encoder: 7-wide conv + 3 dilated residual blocks → D-ch","hw→[B,768,D]   tw(0-pad)→[B,320,D]",COL['trunk'],8.3)
box(0.7,11.75,6.5,1.0,"NL-layer Transformer  (global attention → de-alias GR)","hw→[B,768,D]   tw→[B,320,D]",COL['tx'],8.9)
box(1.5,10.4,4.6,1.0,"Dot-product / √D  →  cost volume C","[B, 768, 320]",COL['cost'],9.2)
box(0.5,8.8,3.7,1.1,"Grounded readout (train-time head)\n2Dconv → softmax → expectation","d_readout [B,768]  (CE+Huber sup.)",COL['read'],8.5)
# hw/tw → shared stem ; hw → context encoder
arr(1.9,14.6,2.3,14.17); arr(5.5,14.6,4.9,14.17)                  # tw→stem, hw→stem
arr(7.0,15.05,8.4,14.5,rad=-0.08)                                # hw → context encoder
arr(3.95,13.15,3.95,12.77); arr(3.95,11.75,3.95,11.42)          # stem→tx→...
arr(3.8,10.4,2.35,9.92,rad=0.12)                                 # cost → readout

# ── RAFT bubble(右下)──
rx,ry,rw,rh=7.0,1.55,6.0,7.65
ax.add_patch(FancyBboxPatch((rx,ry),rw,rh,boxstyle="round,pad=0.02,rounding_size=0.06",fc=COL['raftbg'],ec='#15803d',lw=1.8,ls='--',zorder=1))
ax.text(rx+0.25,ry+rh-0.32,"RAFT iterative refinement",ha='left',fontsize=11,weight='bold',color='#15803d',zorder=4)
box(7.3,7.9,3.0,0.85,"C₀ = pool÷8","[B,96,320] lookup=sinh320",COL['raft'],8.4,7.2)
box(7.3,6.75,4.9,0.85,"Correlation pyramid  (avg_pool×2, 7 levels 320→…→5)",None,COL['raft'],8.2)
lx,ly,lw2,lh=7.3,3.05,4.9,3.35
ax.add_patch(FancyBboxPatch((lx,ly),lw2,lh,boxstyle="round,pad=0.02,rounding_size=0.05",fc='#ffffff',ec='#15803d',lw=1.4,zorder=3))
ax.text(lx+lw2/2,ly+lh-0.26,"iterate loop  ×8 train / ×24 infer",ha='center',fontsize=8.8,weight='bold',color='#15803d',zorder=4)
gru_y=ly+lh-0.62-3*0.50                                          # ConvGRU 那行的 y(context 箭头指它)
steps=["d [B,96] —searchsorted→ frac idx",
       "±4 taps/level → [B, 7×9, 96]",
       "corr-enc [B,64,96] ‖ flow(d/dmax)[B,32,96]",
       "→ motion ⊕ inp → ConvGRU(k9)",
       "→ h[B,H,96] → Δ=head(h) → d←d+Δ"]
for i,s in enumerate(steps):
    ax.text(lx+0.16,ly+lh-0.62-i*0.50,s,ha='left',va='center',fontsize=7.8,family='monospace',color=COL['mono'],zorder=4)
box(7.3,1.75,4.9,0.85,"Convex upsample head","softmax(0.25·conv h) → 3-nbr convex → ×8 → [B,768]",COL['raft'],8.2,6.6)
arr(8.8,7.9,8.8,7.6)                          # C0 → pyramid
arr(9.75,6.75,9.75,6.4)                       # pyramid → loop
arr(9.75,3.05,9.75,2.65)                      # loop → convex
# iterate loop 回环箭头: 末步 d←d+Δ 回喂顶步 d; 弧线锚在白 box 左边缘、向左凸进浅绿气泡(从白 box 外面划过, 不碰框内文字)
ax.add_patch(FancyArrowPatch((lx,ly+0.82),(lx,ly+lh-0.74),arrowstyle='-|>',
             mutation_scale=13,lw=1.9,color='#15803d',connectionstyle="arc3,rad=-0.30",zorder=6))
ax.text(6.62,ly+lh/2-0.1,"d fed back each iter",fontsize=7.4,color='#15803d',ha='center',va='center',rotation=90,weight='bold',zorder=6)
# cost → RAFT
arr(6.1,10.6,7.0,8.45,rad=-0.18,c='#b91c1c',lw=1.9)
ax.text(6.45,9.45,"C (pre-smooth)\n→ RAFT",fontsize=8.2,color='#b91c1c',style='italic',ha='center')
# context encoder → ConvGRU(单竖线走右通道, 明确指向 ConvGRU 那一步)
ax.plot([12.8,12.8],[13.55,gru_y],color='#c2410c',lw=1.8,zorder=5)      # 竖线: context底 → ConvGRU行高
arr(12.8,gru_y,12.25,gru_y,rad=0.0,c='#c2410c',lw=1.8)                  # 左入, 箭头落在 ConvGRU
ax.text(13.15,7.9,"h → ConvGRU init state\ninp → ⊕ motion each iter",fontsize=8.2,color='#c2410c',va='center',ha='left',weight='bold',zorder=5)

# ── 输出(底部)──
box(1.1,0.4,4.3,0.85,"drift (eval)  →  TVT = drift + anchor",None,COL['out'],9.3)
arr(7.3,1.85,5.45,0.95,rad=0.16)                                 # convex → drift(deploy=last round)
arr(2.35,8.8,3.0,1.25,rad=0.06,c='#94a3b8',lw=1.3,ls=':')       # readout → drift(train only)
ax.text(1.45,4.7,"readout: training\nsupervision only",fontsize=7.8,color='#64748b',style='italic',ha='center')
ax.text(7.2,-0.12,"loss = Σ γ^(K−k)·L1 on eval segment;   deploy/OOF = RAFT last round",ha='center',fontsize=8,color='#475569')
ax.text(7.2,-0.52,"D = embed dim (192 grtx-305 / 256 grtx-341) · H = RAFT hidden (96 / 128) · NL = contextualizer layers (5 / 6)",ha='center',fontsize=7.6,color='#64748b',style='italic')
ax.set_title("grtx — shared encoder + context encoder → cost volume → RAFT",fontsize=12.5,weight='bold',pad=6)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'arch_grtx.png'),dpi=145,bbox_inches='tight')
print("saved", os.path.join(OUT,'arch_grtx.png'))
