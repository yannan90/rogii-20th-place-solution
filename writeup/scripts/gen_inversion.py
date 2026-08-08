"""inversion.png — 反演/正演 GR 模拟器(loader2/3 共用)流程图 + 真实 before/after 示例。
调 train_tx.forward_gr。用法: python3 gen_inversion.py"""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _common import OUT, use_train_tx
T = use_train_tx()
D,gm,gs=T.load_wells(); wells=list(D.keys())
w=[x for x in wells if len(D[x]['tvt'])-int(D[x]['split'])>400][0]
wd=D[w]; sp=int(wd['split']); anc=float(wd['tvt'][sp-1]); drift=(wd['tvt']-anc).astype(float)
clean=(np.interp(anc+drift, wd['tw_tvt'], wd['tw_gr'])-T.SYNTH_GMU)/T.SYNTH_GSD
synth=T.forward_gr(drift, anc, wd['tw_tvt'].astype(float), wd['tw_gr'].astype(float), np.random.RandomState(2), sparse=True, split=sp)
xi=np.arange(len(drift))[sp:sp+700]

plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'})
fig=plt.figure(figsize=(14.2,7.4)); gsp=fig.add_gridspec(2,1,height_ratios=[1.05,1.0],hspace=0.32)
axf=fig.add_subplot(gsp[0]); axf.set_xlim(0,14.2); axf.set_ylim(0,3.2); axf.axis('off')
C=['#dCEBFb','#c9d2f7','#fde39a','#fca5a5','#c9e8d3','#bbf7d0']
labs=[("inputs\ndrift trajectory\n+ typewell GR(TVT)",C[0]),("① sample typewell\nalong (anchor+drift)\n= clean GR",C[1]),
      ("② + vertical mismatch\nTVT-persistent field\namp∝signal, per-well k",C[2]),("③ + speckle noise\n(white 0.20)",C[3]),
      ("④ sparsify + interp\n(mimic real GR gaps)",C[4]),("synthetic hw GR",C[5])]
xw=2.03; gap=0.14; x=0.22                                # 6框留右边距, clip_on=False 防裁
for i,(t,c) in enumerate(labs):
    axf.add_patch(FancyBboxPatch((x,1.0),xw,1.5,boxstyle="round,pad=0.02,rounding_size=0.06",fc=c,ec='#1e293b',lw=1.3,clip_on=False))
    axf.text(x+xw/2,1.75,t,ha='center',va='center',fontsize=8.7,weight='bold' if i in(0,5) else 'normal')
    if i<5: axf.add_patch(FancyArrowPatch((x+xw,1.75),(x+xw+gap,1.75),arrowstyle='-|>',mutation_scale=13,lw=1.6,color='#1e293b',clip_on=False))
    x+=xw+gap
axf.set_title("The forward / inversion GR simulator  —  shared by the whip & layer loaders",fontsize=12.5,weight='bold')
axf.text(7.1,0.35,"used every time drift (whip) or formation level (layer) changes: re-generate the GR that well would have logged",
         ha='center',fontsize=8.8,style='italic',color='#475569')
axl=fig.add_subplot(gsp[1])
axl.plot(xi,clean[sp:sp+700],c='#2b6cb0',lw=1.3,label='① clean (typewell sampled along drift)')
axl.plot(xi,synth[sp:sp+700],c='#e53e3e',lw=1.0,alpha=0.85,label='④ final synthetic GR (+ mismatch + noise + sparsify)')
axl.set_xlabel('eval sample'); axl.set_ylabel('normalized GR'); axl.legend(fontsize=9,loc='best'); axl.grid(alpha=0.25)
axl.set_title(f"real example (well {w}): clean → realistic synthetic GR",fontsize=10.5)
fig.savefig(os.path.join(OUT,'inversion.png'),dpi=140,bbox_inches='tight'); print("saved", os.path.join(OUT,'inversion.png'))
