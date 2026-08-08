# writeup/ — ROGII Gold Solution 分享稿

## 内容
- **`writeup.md`** — 正文(英文, 论坛分享稿)。
- **`*.png`** — 正文引用的 10 张图。
- **`scripts/`** — 生图脚本(除 3 张竞赛期原图外, 每张图都能一键复现)。
- **`OUTLINE.md`** — 中文提纲工作稿(讨论用, 非最终稿)。

## 图 → 脚本对照

| 图 | 出处 | 说明 |
|---|---|---|
| `arch_grtx.png` | `scripts/gen_arch_grtx.py` | grtx 架构(双 encoder→cost volume→RAFT), 自包含 |
| `arch_pfnet.png` | `scripts/gen_arch_pfnet.py` | pfnet 双分支 residual-on-base, 自包含 |
| `cv_vs_lb.png` | `scripts/gen_cv_vs_lb.py` | CV/pub/priv 散点(提交策略), 数据硬编码自提交历史 |
| `raft_vs_readout.png` | `scripts/gen_raft_vs_readout.py` | RAFT vs readout 逐井增益/损害(tx341 OOF), 自包含 |
| `aug_shape.png` | `scripts/gen_aug_families.py` | shape 换形状(6井+统计), **需 train_tx** |
| `aug_kink.png` | `scripts/gen_aug_families.py` | kink 折角注入(6井+统计), **需 train_tx** |
| `aug_layer.png` | `scripts/gen_aug_families.py` | layer 层位平移(6井+统计), **需 train_tx** |
| `inversion.png` | `scripts/gen_inversion.py` | 反演/正演 GR 模拟器流程+示例, **需 train_tx** |
| `sinh_grid.png` | 竞赛期原图(无脚本) | 均匀 vs sinh 状态网格 |
| `real_vs_whip.png` | 竞赛期原图(无脚本) | 真井 vs whip 合成 drift 漏斗 |
| `viz_landed.png` | 竞赛期原图(无脚本) | landing-aware anchor 滑窗范围 |

## 复现

**自包含图**(无需数据/train_tx, 任意目录可跑):
```bash
cd writeup/scripts
python3 gen_arch_grtx.py
python3 gen_arch_pfnet.py
python3 gen_cv_vs_lb.py
```

**需 train_tx 的图**(shape/kink/layer/inversion): 依赖项目根 `train_tx.py` + 官方数据 + `CV_splits/folds.json` + `data/true_drift_cache.npz`。脚本内 `_common.use_train_tx()` 会自动定位项目根并 `chdir` 过去再 import(train_tx 按项目根解析数据路径), 所以从 `scripts/` 里直接跑即可:
```bash
cd writeup/scripts
python3 gen_aug_families.py     # → aug_shape/kink/layer.png
python3 gen_inversion.py        # → inversion.png
```

## 依赖
- `numpy`, `matplotlib`(全部脚本)
- 需 train_tx 的脚本额外需要: 项目根可 import `train_tx`, 且官方数据/缓存就位(同训练环境)。

## 备注
- `scripts/_common.py` = 公共路径工具(定位项目根 / 输出到 writeup/ / 挡 train_tx 的 argparse)。
- 图输出统一写到 `writeup/`(覆盖同名文件)。
- 3 张竞赛期原图(sinh_grid / real_vs_whip / viz_landed)在竞赛过程中产出, 生成脚本未随本 writeup 固化。
