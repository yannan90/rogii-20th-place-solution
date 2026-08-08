"""arch_pfnet.png — pfnet(net209)架构: gen_gpupf 同源产出(1D PF base + 2D traj)→ 双分支 residual-on-base。
自包含 schematic, 无外部依赖。用法: python3 gen_arch_pfnet.py"""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _common import OUT
plt.rcParams.update({'font.size':9.5,'font.family':'DejaVu Sans'})
W,H=15.7,7.6
fig,ax=plt.subplots(figsize=(13.6,6.6)); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')
C=dict(pf='#c7d2fe',base='#dCEBFb',traj='#c9d2f7',feat='#bcd6f5',combo='#fde39a',
       b1='#bcd6f5',b2='#c9e8d3',sum='#fca5a5',out='#bbf7d0')
def box(x,y,w,h,title,shape=None,fc='#eee',fs=9.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc=fc,ec='#1e293b',lw=1.3,zorder=3))
    if shape:
        ax.text(x+w/2,y+h*0.64,title,ha='center',va='center',fontsize=fs,weight='bold',zorder=4)
        ax.text(x+w/2,y+h*0.27,shape,ha='center',va='center',fontsize=fs-1.3,family='monospace',color='#b91c1c',zorder=4)
    else:
        ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=fs,weight='bold',zorder=4)
def arr(x1,y1,x2,y2,rad=0,c='#1e293b',lw=1.7):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,lw=lw,color=c,connectionstyle=f"arc3,rad={rad}",zorder=2))

# ── 源: gen_gpupf(同源产出两读出; 部署 net209 base 配置)──
ax.add_patch(FancyBboxPatch((0.3,4.35),3.2,2.2,boxstyle="round,pad=0.02,rounding_size=0.06",fc=C['pf'],ec='#1e293b',lw=1.3,zorder=3))
ax.text(1.9,6.28,"gen_gpupf",ha='center',fontsize=9.6,weight='bold',zorder=4)
ax.text(1.9,5.95,"GPU particle filter",ha='center',fontsize=8.6,zorder=4)
for j,ln in enumerate(["gs_mult=3 · α=0.998","w_geo=0.05 · SDE motion","64 seeds × 500 particles"]):
    ax.text(1.9,5.55-j*0.34,ln,ha='center',fontsize=7.9,family='monospace',color='#3730a3',zorder=4)
# ── 输入(左下)──
box(0.3,2.95,3.2,1.0,"hw features","[B, 14, n_ev]  GR/geom-derived",C['feat'],9.1)
box(0.3,1.45,3.2,1.0,"combo channel","[B, n_ev]  6-layer formation kriging",C['combo'],8.9)
# ── gen_gpupf 的两个 readout(同源)──
box(4.3,5.55,3.15,0.95,"PF base — geo_lik (1D)","[B, n_ev]  OOF 9.77",C['base'],9.0)
box(4.3,4.25,3.15,0.95,"2D traj image","[B, 64, n_ev]  64 particle traj",C['traj'],9.0)
arr(3.5,5.9,4.3,6.02,rad=-0.05); arr(3.5,5.2,4.3,4.72,rad=0.10)
ax.text(5.87,6.75,"gen_gpupf dual readout  (same PF run → 1D base + 2D traj)",ha='center',fontsize=8.0,style='italic',color='#4338ca',zorder=4)
# ── 双分支(带结构)──
def branch(x,y,w,h,title,lines,shape,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc=fc,ec='#1e293b',lw=1.3,zorder=3))
    ax.text(x+w/2,y+h-0.28,title,ha='center',fontsize=9.1,weight='bold',zorder=4)
    for i,ln in enumerate(lines):
        ax.text(x+w/2,y+h-0.62-i*0.34,ln,ha='center',fontsize=7.7,family='monospace',color='#334155',zorder=4)
    ax.text(x+w/2,y+0.24,shape,ha='center',fontsize=8.0,family='monospace',color='#b91c1c',zorder=4)
branch(8.2,2.35,3.35,1.55,"1-D residual branch",
       ["conv1d residual stack","(3× k5 dilated, 128ch)"],"r₁ [B, n_ev]",C['b1'])
branch(8.2,4.55,3.35,1.55,"2-D branch — UNet",
       ["enc–dec 32ch over the","[64 × n_ev] traj heatmap"],"r₂ [B, n_ev]",C['b2'])
arr(3.5,3.35,8.2,3.15,rad=0.02)                         # features → 1D
arr(3.5,1.9,8.2,2.8,rad=0.10)                           # combo → 1D
arr(7.45,4.72,8.2,5.2,rad=0.06)                         # traj → 2D
# ── 求和 + 输出 ──
box(12.3,3.55,2.2,1.4,"sum\ndrift = PF base\n+ r₁ + r₂",None,C['sum'],9.2)
box(14.9,3.8,0.7,0.9,"drift","~8.67",C['out'],9.0)
# PF base → sum: elbow(顶部平走过 2D 框【上方】, 到 sum 前再下折入顶, 不擦任何框)
ax.plot([7.45,12.62],[6.5,6.5],color='#1d4ed8',lw=2.0,zorder=2)
arr(12.62,6.5,12.62,4.98,rad=0,c='#1d4ed8',lw=2.0)      # 下折入 sum 顶
ax.text(9.7,6.72,"PF base carried through  (residual-on-base)",fontsize=8.3,color='#1d4ed8',style='italic',ha='center',zorder=4)
arr(11.55,3.1,12.3,3.95,rad=0.10)                       # 1D → sum
arr(11.55,5.2,12.3,4.55,rad=-0.10)                      # 2D → sum
arr(14.5,4.25,14.9,4.25)                                # sum → drift
ax.set_title("pfnet (net209) — two-branch residual on a strong GPU particle-filter base",fontsize=12.5,weight='bold',pad=6)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'arch_pfnet.png'),dpi=145,bbox_inches='tight')
print("saved", os.path.join(OUT,'arch_pfnet.png'))
