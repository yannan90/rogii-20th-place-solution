"""raft_vs_readout.png — 用 tx341_5fold OOF 分析 RAFT 相对 readout 的逐井增益/损害。
自包含(只读 npz)。用法: python3 gen_raft_vs_readout.py"""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import ROOT, OUT
def L(p): z=np.load(os.path.join(ROOT,p),allow_pickle=True); return {k:z[k] for k in z.files}
TRUE=L('data/true_drift_cache.npz')
RAFT=L('experiments/gr_tx/tx341_5fold/oof_gr_tx.npz')        # 部署 = RAFT 末轮
RO  =L('experiments/gr_tx/tx341_5fold/readout_cache.npz')    # readout(pre-RAFT)
def al(a,n): a=np.asarray(a,float); return a if len(a)==n else np.interp(np.linspace(0,1,n),np.linspace(0,1,len(a)),a)
ws=[w for w in TRUE if w in RAFT and w in RO]
rmse=lambda p,t: float(np.sqrt(np.mean((p-t)**2)))
r_ro=[]; r_rf=[]
for w in ws:
    t=TRUE[w].astype(float); n=len(t)
    r_ro.append(rmse(al(RO[w],n),t)); r_rf.append(rmse(al(RAFT[w],n),t))
r_ro=np.array(r_ro); r_rf=np.array(r_rf); delta=r_rf-r_ro       # <0 = RAFT 更好
imp=delta< -1e-6; hurt=delta>1e-6
# pooled
Y=np.concatenate([TRUE[w].astype(float) for w in ws])
P_ro=np.concatenate([al(RO[w],len(TRUE[w])) for w in ws]); P_rf=np.concatenate([al(RAFT[w],len(TRUE[w])) for w in ws])
pool_ro=rmse(P_ro,Y); pool_rf=rmse(P_rf,Y)
print(f"wells={len(ws)}  improved={imp.sum()}  hurt={hurt.sum()}  pooled readout={pool_ro:.3f} raft={pool_rf:.3f}")
print(f"mean gain on improved={-delta[imp].mean():.2f}ft  mean damage on hurt={delta[hurt].mean():.2f}ft")

plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(1,2,figsize=(12.5,5.2))
# 散点: readout vs raft per-well RMSE
mx=max(r_ro.max(),r_rf.max())*1.02
ax[0].plot([0,mx],[0,mx],'--',c='#64748b',lw=1.3,zorder=1)
ax[0].scatter(r_ro[imp],r_rf[imp],s=22,c='#2e9e5b',alpha=0.8,zorder=3,label=f'RAFT better ({imp.sum()})')
ax[0].scatter(r_ro[hurt],r_rf[hurt],s=22,c='#e45756',alpha=0.8,zorder=3,label=f'RAFT worse ({hurt.sum()})')
ax[0].set_xlabel('readout per-well RMSE (ft)'); ax[0].set_ylabel('RAFT per-well RMSE (ft)')
ax[0].set_title(f'RAFT vs readout — per well (grtx-341)\nbelow diagonal = RAFT helps',fontsize=11)
ax[0].legend(fontsize=9,loc='upper left'); ax[0].grid(alpha=0.25,zorder=0); ax[0].set_xlim(0,mx); ax[0].set_ylim(0,mx)
# delta 直方
ax[1].hist(np.clip(delta,-6,6),bins=40,color='#4c78a8',alpha=0.85,edgecolor='white')
ax[1].axvline(0,c='#111',lw=1.2); ax[1].axvline(np.median(delta),c='#e45756',lw=2,label=f'median {np.median(delta):+.2f}ft')
ax[1].set_xlabel('Δ RMSE = RAFT − readout (ft)   ← RAFT better | RAFT worse →')
ax[1].set_ylabel('# wells'); ax[1].set_title(f'per-well Δ  (pooled: readout {pool_ro:.2f} → RAFT {pool_rf:.2f})',fontsize=11)
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25)
fig.suptitle(f'RAFT refinement over the readout: helps {imp.sum()} wells, hurts {hurt.sum()} (mostly the hard ones)',
             fontsize=12.5,weight='bold')
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(os.path.join(OUT,'raft_vs_readout.png'),dpi=135,bbox_inches='tight')
print("saved", os.path.join(OUT,'raft_vs_readout.png'))
