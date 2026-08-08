"""cv_vs_lb.png — 单 grtx 模型的 本地CV / public LB / private LB 散点(提交策略杀手图)。
自包含(数据硬编码自提交历史), 无外部依赖。用法: python3 gen_cv_vs_lb.py"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _common import OUT

# grtx 自研单模: (name, cv, public_lb, private_lb)
d = [("305",6.408,5.083,7.303),("337",6.090,5.305,7.092),("341",5.942,5.439,6.691),
     ("346",6.070,5.360,6.867),("379",6.413,5.242,6.897),("388",6.780,5.150,6.956),
     ("389",6.625,5.231,7.010),("424",6.095,5.536,6.830),("468",6.439,6.051,7.075)]
cv=np.array([x[1] for x in d]); pub=np.array([x[2] for x in d]); priv=np.array([x[3] for x in d])
r_cv=np.corrcoef(cv,priv)[0,1]; r_pub=np.corrcoef(pub,priv)[0,1]
print(f"corr(CV, priv)={r_cv:+.3f}  corr(pubLB, priv)={r_pub:+.3f}")

fig,ax=plt.subplots(1,2,figsize=(11,4.6))
for a,(x,lab,r) in zip(ax,[(cv,'Local CV (OOF)',r_cv),(pub,'Public LB (50 wells)',r_pub)]):
    a.scatter(x,priv,s=70,c='#2b6cb0',zorder=3,edgecolor='white',linewidth=1.2)
    for xi,yi,nm in zip(x,priv,[q[0] for q in d]):
        a.annotate(nm,(xi,yi),fontsize=8,xytext=(4,4),textcoords='offset points',color='#444')
    m,b=np.polyfit(x,priv,1); xs=np.linspace(x.min(),x.max(),50)
    a.plot(xs,m*xs+b,'--',c='#e53e3e',lw=1.5,zorder=2)
    a.set_xlabel(lab); a.set_ylabel('Private LB (150 wells)')
    a.set_title(f'{lab} vs Private   (Pearson r = {r:+.2f})',fontsize=11); a.grid(alpha=0.25,zorder=0)
fig.suptitle('Single grtx models: what predicts Private LB?  Local CV wins, Public LB is noise',
             fontsize=12,weight='bold')
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(os.path.join(OUT,'cv_vs_lb.png'),dpi=130,bbox_inches='tight')
print("saved", os.path.join(OUT,'cv_vs_lb.png'))
