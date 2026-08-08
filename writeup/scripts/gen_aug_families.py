"""aug_shape.png / aug_kink.png / aug_layer.png — 三个 loader 的增强样本图(多井 + 统计)。
调 train_tx 真实合成函数(make_shape_well / make_layer_well)。
⚠️ 需从项目根解析数据(_common.use_train_tx 已 chdir)。用法: python3 gen_aug_families.py"""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import OUT, use_train_tx
T = use_train_tx()
T.AUG_SLIDE=False; T.SHAPE_AUG_P=1.0                    # demo: 固定split干净对比 + 必增强
D,gm,gs=T.load_wells(); wells=list(D.keys())
shape_lib,shape_max=T.build_shape_lib(D,wells)
def evd(wd,sp,anc): return wd['tvt'][sp:]-anc

# 6 口示例井 + 200 口统计池
demo=[]
for w in wells:
    wd=D[w]; sp=int(wd['split']); n=len(wd['tvt'])-sp; a=wd['ancc'][sp:]
    if n>300 and np.all(np.isfinite(a)) and (wd['tvt'][sp:]-wd['tvt'][sp-1]).ptp()>12: demo.append(w)
    if len(demo)>=6: break
pool=[]
for w in wells:
    wd=D[w]; sp=int(wd['split']); n=len(wd['tvt'])-sp; a=wd['ancc'][sp:]
    if n>250 and np.all(np.isfinite(a)): pool.append(w)
    if len(pool)>=200: break

sh_rms=[]; kk_disp=[]; ly_shift=[]
for w in pool:
    wd=D[w]; sp=int(wd['split']); anc=float(wd['tvt'][sp-1]); dr=evd(wd,sp,anc)
    T.SHAPE_KINK_P=0.0; np.random.seed(hash(w)%9999)
    d0=T.make_shape_well(dict(wd),shape_lib,shape_max)['tvt'][sp:]-anc; sh_rms.append(np.sqrt(np.mean((d0-dr)**2)))
    T.SHAPE_KINK_P=1.0; np.random.seed(hash(w)%9999)
    d1=T.make_shape_well(dict(wd),shape_lib,shape_max)['tvt'][sp:]-anc; kk_disp.append(np.max(np.abs(d1-d0)))
    np.random.seed(hash(w)%9999)
    ly_shift.append(abs(float(T.make_layer_well(dict(wd))['tvt'][sp:].mean()-wd['tvt'][sp:].mean())))
sh_rms=np.array(sh_rms); kk_disp=np.array(kk_disp); ly_shift=np.array(ly_shift)

def panels(fn,title,gen,ylab='drift (ft)',stat=None,statlab=''):
    fig=plt.figure(figsize=(14,6.2)); gs_=fig.add_gridspec(2,4,width_ratios=[1,1,1,1.15])
    for i,w in enumerate(demo):
        ax=fig.add_subplot(gs_[i//3,i%3]); wd=D[w]; sp=int(wd['split']); anc=float(wd['tvt'][sp-1])
        xi=np.arange(len(wd['tvt'])-sp); a,b,la,lb,ca,cb=gen(wd,sp,anc)
        ax.plot(xi,a,c='#2b6cb0',lw=1.6,label=la); ax.plot(xi,b,c=cb,lw=1.6,label=lb)
        ax.set_title(w,fontsize=9); ax.grid(alpha=0.22)
        if i==0: ax.legend(fontsize=8,loc='best')
        if i%3==0: ax.set_ylabel(ylab,fontsize=8.5)
    axs=fig.add_subplot(gs_[:,3]); axs.hist(stat,bins=22,color='#4c78a8',alpha=0.85,edgecolor='white')
    axs.axvline(np.median(stat),c='#e45756',lw=2,label=f'median {np.median(stat):.1f}')
    axs.set_title(statlab,fontsize=10); axs.set_xlabel('ft'); axs.set_ylabel('# wells'); axs.legend(fontsize=8.5); axs.grid(alpha=0.2)
    fig.suptitle(title,fontsize=13,weight='bold'); fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(OUT,fn),dpi=135,bbox_inches='tight'); plt.close(fig); print("saved",fn)

def g_shape(wd,sp,anc):
    dr=wd['tvt'][sp:]-anc; T.SHAPE_KINK_P=0.0; np.random.seed(hash(wd['tvt'][0].tobytes())%9999)
    s=T.make_shape_well(dict(wd),shape_lib,shape_max); return dr,s['tvt'][sp:]-anc,'real drift','shape-swapped','#2b6cb0','#e53e3e'
panels('aug_shape.png','Shape-library swap — new ANCC shape per sample (drift, 6 wells)',g_shape,
       stat=sh_rms,statlab=f'drift-shape change (RMS, {len(pool)} wells)')

# ── kink 图: 直接画 ANCC 曲线, 标天然 kink(曲率峰) vs 注入 kink ──
T.SHAPE_ALPHA_MIN=0.0; T.SHAPE_ALPHA_MAX=0.0; T.SHAPE_KINK_STEER=False   # 只注入kink(不换形状/不进Z)→ 合成ANCC=真ANCC+kink
kdemo=demo[:4]
figk,axk=plt.subplots(2,2,figsize=(12,7.2))
for i,w in enumerate(kdemo):
    ax=axk[i//2,i%2]; wd=D[w]; sp=int(wd['split']); n=len(wd['tvt'])-sp; xi=np.arange(n)
    real=np.asarray(wd['ancc'][sp:],float)
    T.SHAPE_KINK_P=1.0; np.random.seed(hash(w)%9999)
    syn=T.make_shape_well(dict(wd),shape_lib,shape_max)
    synA=real+(np.asarray(syn['tvt'][sp:],float)-np.asarray(wd['tvt'][sp:],float))   # 注入kink后的ANCC
    sm=np.convolve(real,np.ones(15)/15,mode='same'); curv=np.abs(np.gradient(np.gradient(sm)))
    m=max(int(0.1*n),5); curv[:m]=0; curv[-m:]=0; nk=int(np.argmax(curv)); ik=int(np.argmax(np.abs(synA-real)))
    ax.plot(xi,real,c='#2b6cb0',lw=1.7,label='real ANCC'); ax.plot(xi,synA,c='#dd6b20',lw=1.5,ls='--',label='+ injected kink')
    ax.plot(nk,real[nk],'v',c='#1d4ed8',ms=12,zorder=5,label='natural kink'); ax.plot(ik,synA[ik],'*',c='#c2410c',ms=16,zorder=5,label='injected kink')
    ax.set_title(w,fontsize=9); ax.grid(alpha=0.22); ax.set_ylabel('ANCC (formation TVT)',fontsize=8); ax.set_xlabel('eval sample',fontsize=8)
    if i==0: ax.legend(fontsize=7.8,loc='best')
figk.suptitle('Kink injection — sharp formation bends in the ANCC curve (natural ▼ vs synthetic ★)',fontsize=12.5,weight='bold')
figk.tight_layout(rect=[0,0,1,0.96]); figk.savefig(os.path.join(OUT,'aug_kink.png'),dpi=135,bbox_inches='tight'); plt.close(figk); print("saved aug_kink.png")

def g_layer(wd,sp,anc):
    np.random.seed(hash(wd['Z'][0].tobytes())%9999); ly=T.make_layer_well(dict(wd))
    return wd['grraw'][sp:],ly['grraw'][sp:],'real GR','layer-shifted GR','#2b6cb0','#38a169'
panels('aug_layer.png','Layer shift — bulk formation move → GR changes, drift fixed (GR, 6 wells)',g_layer,
       ylab='GR',stat=ly_shift,statlab=f'applied layer shift |Δ| ({len(pool)} wells)')
