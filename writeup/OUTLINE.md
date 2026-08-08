# ROGII Wellbore Geology Prediction — Gold Solution Writeup(中文提纲 v1)

> 简版提纲。数字 🟡=待你从提交历史确认。后续再延展/配图/翻译。

https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/
查一下主办方网站，说感谢主办方，感谢kaggle community。这个竞赛非常有意思！非常复杂的问题。可以尝试的方法非常多。之前很多竞赛我经常试了很多方法都没有用最后回归到一个很简单的方法。
但是这次不是，这次真的是more is more。 发现了很多有用的有意思的方法。最后我的主力走的是一个深度神经网络路线。接下来细细展开。

---

## 1. Big Picture:多路去相关 ensemble

一句话:**没有单一模型能赢,赢的是三条机制互不相关的路的融合。**

三条路:
- **主力:gr_tx** —— 单井 GR↔typewell 迭代配准 transformer(纯单井信息,无邻井)
- **物理路:pfnet + tree** —— 粒子滤波 base + 残差学习(GR + 空间先验的物理融合)
- **几何路:kriging + geom_tx** —— 不碰 GR,纯几何/邻井地质外推(和 gr_tx 机制正交)

融合方式(逐点):
```
out = (1 − w_pf − w_geo_ancc − w_geo_geomtx) · gr_tx
      + w_geo_ancc · kriging  + w_geo_geomtx · geom_tx   ← 几何路
      + w_pf · (pfnet / tree)                            ← 物理路
```
**权重分配的关键 = 几何路是【距离变权】的**:`w = W / (1 + (nd/D)²)`,nd=该点到最近邻井 ANCC 控制点的距离。
- 稠密井区(nd 小)→ 几何腿权重高(邻井地质可信);孤立井(nd 大)→ 权重衰减到 0(不信外推)。
- ancc 快衰减(D≈250–700)、geom_tx 慢衰减(D≈1200–1500):ancc 强在近距、geom_tx 强在中距,分工交叉点 nd≈300。
- pfnet/tree 是固定权重(逐井/逐点门控撞 selection 墙,见负结果)。
- 🟡 最终配比:待填(gr_tx 吃残余,ancc~0.02–0.05 / geom~0.04 / pf~0.10–0.15)。

---

## 2. gr_tx —— 主力(可讲的最多)

(这里提一下后面提到模型编号的时候用grtx-xxx表示，方便理解)

### 2.1 数据 / 特征处理
- **cv策略**： 五折随机，每次lb提交都直接提交五组
- **维度挑选**:水平井 GR 序列压到 `COMPRESSION=16ft/点`(U 形甜点:12ft/24ft 都更差)、`N=768` 点;typewell `S=320` 状态;`INPUT_CLIP=80`(known 段只保留 anchor±80ft,远端 known 不相关反伤)。
- **sinh 非均匀状态网格**(🔑核心):drift 概率质量高度集中(p50=8ft/p90=25/p100=104),均匀 ±60 网格既浪费中心又罩不住尾。改 `g = A·sinh(b·t)` 非均匀网格:中心 0.26ft/格(细)、尾部到 ±115ft(全覆盖)。解析可逆(闭式 asinh 定位,免二分)。
- **per-well 联合鲁棒归一**:hw+tw 拼一起算【一个】median/MAD 同除 → 保住 hw/tw 增益关系、只去 per-pair 整体 offset/scale(减记忆井身份)。

### 2.2 架构:双 encoder + transformer + RAFT
- **双 encoder(hw/tw 跨模态)**:hw(脏 LWD + ΔZ + MD 轴 16ft/点)vs tw(干净 GR + TVT sinh 轴)—— 物理尺度本就不同,conv stem 编码。
- **transformer contextualizer**(NL=5–6,DIM 192–256):全局 context 消 GR 别名歧义(GR 非单射,局部易配错,全局 attention 去歧义)。
- **grounded readout**:cross-attention 出 correspondence(代价图 C),`drift = correspondence @ ref_drift`;`CE_W=10` 直接监督 C 在真 drift 状态处峰化;2D 卷积正则代价图(stereo/flow 式)。
- **RAFT 迭代精修**:相关金字塔(7 层,每轮 ±4 taps 查表覆盖全 ±115 状态轴),迭代更新 drift(K9=9 轮);**部署 = RAFT 末轮**。

💊需要做架构图。画完补充到writeup文件夹里。

### 2.3 数据合成引擎(破 773 井过拟合的命门)
773 井太少 → 匹配模型卡天花板。正演造无限训练对:
- **真井增强(SHAPE)**:把真井 ANCC 形状换成形状库里的形状(α 混合、方差守恒);+ **kink 折角注入**(难井第一难度轴=折角,人口基率才 2–3% = 真缺口);+ **司钻响应(steer)**:折角处 ΔZ 滞后追(真井 ΔZ 跟弯比 0.50,不装瞎)。
- **whip(鞭子漏斗)**:随机 drift 轨迹(平台+起伏+趋势多风格),漏斗式张开。
- **layer(层位平移)**:整体层位平移增广。
- **forward GR + 失配**:typewell 沿 drift 正演 GR + vp-only 失配(TVT 持久垂直局部失配,非 warp,全标定自真实 773 井)+ 毛刺噪 + 抽稀。

💊把这几种增强都画一些图，可视化一下。

---

## 3. 物理路:pfnet + tree

(这里提一下后面提到模型编号的时候用pfnet-xxx, pftree-xx表示，方便理解)

### 3.1 PF 的 GPU 化(gen_gpupf)
- 统一 GPU 粒子滤波:一个函数 `gen_gpupf(Z, **10参数)` → {井:drift},train 准备 + kernel 推理统一调用。
- 10 参数(seeds/particles/alpha 动量/SDE motion/发射宽度/kriging geo 先验 w_geo/readout…)→ 一套代码生成所有 PF 专家。
- 强 base:`gpupf_g3geo_krig`(w_geo0.05 kriging 先验)裸 OOF **9.77 ≪ 竞品自带弱 PF 14.35**。

### 3.2 pfnet(net209)架构
- 基座 = nn114_twin:**双分支** = geo_lik(1D PF base 残差)+ TRAJ2D(g3 粒子轨迹 2D UNet 分支)+ combo(6 层地层 kriging 通道)。
- **residual-on-PF-base**:PF base 打底 + net 学残差(仿地质物理:base 给大势,net 修局部)。
- **twin 增广**:forward-sim 孪生井(eval 段 GR 换 typewell 正演,提泛化)。单模 ~8.67。

💊模型架构可能需要出个图，可视化一下。

### 3.3 tree(SP45 表格残差)特征研究
- 基座 195 特征(扒自竞品 SP45):48 地层面跨井 kriging + PF + NCC + beam + GR 滚动。
- **特征研究三层**:
  - **dense6**:6 层稠密 IDW imputer(LOO 排自井)= 几何互补主力。
  - **conjb**:GR 合取匹配多描述子(typewell 参考 / known 段自参考 / 密集 kfit 搜索)。
  - **residual base**:gpupf 强 PF 当 base(−1.5 大杠杆)/ 或 net209 当 base(net 升级版)。
- 模型池:xgb + 深度多样 cat(d7/d4)→ ridge 堆叠。

---

## 4. 几何路(简讲带过)
- **部署 kriging**:ANCC 地层面空间 IDW(K60,LOO 排自井),drift base ~14.6。(这里提一下后面提到集成模型编号的时候用ancc，方便理解)
- **geo_tx**:GR-free 几何 transformer,9 通道(known drift 历史 / ΔZ / dx,dy / heading / grid-kriging 邻井地质 / nn_dist 可靠度),滑窗增广。单模 ~11.3。(这里提一下后面提到集成模型编号的时候用geotx，方便理解)
- 共性:单模都卡 11–14ft(地质有 GR 才看得见的不可约成分),但**机制和 gr_tx 完全正交 → blend 大增益**。

💊需要geom_tx需要画一个简单的架构图？或者直接文字/代码就能描述清除？你自己判断一下。画完补充到writeup文件夹里。

---

## 5. CV–LB 分数演进 + 提交策略(⭐重点)

### 5.1 分数演进（这里强调说的都是pub lb，但是pub lb在小的幅度上有一定欺骗性，后面展开讲）
- 🟡 单 gr_tx:LB **5.08**(合成引擎是从"匹配天花板"到上榜的最大杠杆)。
- 🟡 + 几何路(ancc):5.08 → **5.03**。
- 🟡 + 物理路(PF):→ **4.966**。
- **核心规律:去相关 > 单模强度。** 铁证:residual-on-net 的 tree 单模更强(8.44)但和 net209 相关 0.96 → blend 零增益;而弱的 GR-free 几何腿(单模才 11–14)进 blend −0.4。加去相关腿,永远比强化单模值钱。

### 5.2 这次最大的陷阱:public LB 不可信(⭐⭐提交策略)
- **public LB 只有 50 井 → 噪声极大**(50 井 pooled σ≈0.9,两模型差值 σ≈0.6),不足以做模型选择。
- **我的亲身教训**:
  - 我 **public LB 最高的那个 sub 根本没选** —— 它是单模,本地 CV 一般。
  - 我选了两个 **ensemble 更多模型** 的 sub,它们的 **CV 和 public LB 呈负相关**(CV 更高但 public 更低)。
  - 结果 **private LB(150 井)反而和 CV 一致** —— 选 CV 更高/public 更低的 sub 是对的。
  - **甚至:如果我更坚决地选 CV 最高那个(public 更低),private 会更好。**
- **结论:相信本地 CV,别追 public LB。** public 是 50 井的一次抽样,private 150 井才接近真分布;用 public 选模型 = 被噪声牵着走。

- 图表：grtx自研
model | cv | pub lb | priv lb
grtx-305 | 6.408 | 5.083 | 7.303
grtx-337 | 6.090 | 5.305 | 7.092
grtx-341 | 5.942 | 5.439 | 6.691
grtx-346 | 6.070 | 5.360 | 6.867
grtx-379 | 6.413 | 5.242 | 6.897
grtx-388 | 6.780 | 5.150 | 6.956
grtx-389 | 6.625 | 5.231 | 7.010
grtx-424 | 6.095 | 5.536 | 6.830
grtx-468 | 6.439 | 6.051 | 7.075

这里可以画两个点阵图，cv-privlb, publb-privlb, 然后计算一下两者的相关性，得出cv-privlb远好。

- 图表：最终提交策略
strategy | cv | pub lb | priv lb | selected
grtx-305 + pfnet-209*0.15 + ancc*0.05@250 + geotx0.05@1200 | 6.03 | 4.902 | 6.914 | no
grtx-[305+379+388+389] + pfnet-209*0.15 + ancc*0.02@700 + geotx0.04@1500| 5.774 | 4.939 | 6.639 | yes
grtx-[305+337+341+346+424+468] + pfnet-209*0.10 + tree083*0.05 + ancc*0.05@250 + geotx0.05@1200 | 5.437 | 5.197 | 6.500 |yes
grtx-[337+341+346+424+468] + pfnet-209*0.10 + pftree-083*0.05 + ancc*0.10@250 | 5.398 | 5.299 | 6.424 | no

这里结论讲一下，如果我选了publb最高的，我就直接toasted掉出金牌区了。幸好没有被overfitted pub lb 蒙蔽双眼

---

## 6. 负结果目录(和正结果一样值钱)
- **后处理是墓地**:offset/centering/tilt、小几何"向面拉"、逐点/逐井门控(best_w/CONF_GATE)、ΔZ 校正、幅度收缩 —— **OOF 全绿却 LB 翻**。根因同 5.2:隐藏井 + 榜噪声,确定性微调 = 对不利子集的精确测量。
- **GR 非单射 = 不可约尾部**:难井真值的 GR 匹配客观上劣于一条错层 spurious warp → 任何 decoder(RAFT/DP/后处理)被同样带偏。误差集中在极端难井,逐点利用撞 selection 墙。
- **空间 CV 伤 LB**、in-context/邻井 offset 死路 等。

---
## 待办
- 🟡 填 5.1 确切 LB 阶梯 + 最终提交配比 + 最终 private 名次/分数。
- 图:① blend 三路结构图 ② sinh 网格 vs 均匀网格 ③ 合成井 vs 真井对比 ④ CV–public LB 负相关散点(提交策略的杀手图)。
- 延展各节 → 正式 prose → 英文翻译。
