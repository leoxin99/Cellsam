# Adapter 与 CellSAM 数据集技术参考 (A1/Codex, 2026-03-04)

> 目的: 统一沉淀两类信息  
> 1) 本项目 Adapter 相关技术细节 (用于后续实现/复现实验)  
> 2) CellSAM 论文的数据集口径 (用于论文写作时避免误写)

---

## 1. 本项目 Adapter: 结构、流程、结论

### 1.1 Adapter 在我们项目中的位置

当前训练入口在 `src/train.py`:

1. 在 `create_model()` 中按配置创建 adapter (`use_adapter=true`)
2. 在 `create_optimizer()` 中把 adapter 参数加入优化器
3. 在 `train_one_epoch()` / `validate()` 中对输入图像先执行 `images = adapter(images)`，再进入 CellSAM 主干

代码定位:
- `src/train.py:205`
- `src/train.py:230`
- `src/train.py:292`
- `src/train.py:576`

### 1.2 Adapter 结构

当前实现文件: `src/adapters/channel_adapter.py`

两种实现:

1. `IndependentChannelAdapter` (主方案)
- 每个通道独立 3x3 卷积 + bias
- 默认恒等初始化 (中心=1, 其余=0)
- 参数量约 30

2. `LightweightChannelAdapter` (备选)
- 每通道 gain+bias
- 参数量 6

代码定位:
- `src/adapters/channel_adapter.py:20`
- `src/adapters/channel_adapter.py:96`

### 1.3 Adapter 与三通道映射关系

Adapter 不负责“生物通道语义映射”，语义映射由 `SemanticChannelMapper` 完成:

- 旧映射 (T18/T28): `R=BF, G=Actn2, B=DAPI`
- 官方映射对齐分支 (T29): `R=blank/Actn2, G=DAPI, B=BF`

代码定位:
- `src/augmented_dataset.py:25`
- `src/augmented_dataset.py:62`
- `src/augmented_dataset.py:419`

### 1.4 已有实验结论 (当前可引用口径)

在已记录结果中，3ch 有净收益，但 adapter 本身贡献不显著:

- T18-A/B/C 汇总显示: 3ch 平均优于 BF 控制组
- 但 `3ch + adapter` 与 `3ch no-adapter` 非显著差异

记录来源:
- `docs/a2_handoff_20260225.md:48`
- `docs/a2_handoff_20260225.md:50`
- `docs/a2_handoff_20260225.md:52`
- `docs/a2_handoff_20260225.md:63`

> 写作建议: 把 Adapter 表述为“已验证的可插拔模块”，不要写成“主要增益来源”。

---

## 2. CellSAM Methods 复核 (Nature 2025 论文口径)

论文来源:
- Nature article: https://www.nature.com/articles/s41592-025-02879-w
- Nature PDF: https://www.nature.com/articles/s41592-025-02879-w.pdf

### 2.1 数据集与数据构建 (Methods)

按 Nature 论文 Methods (本地 `docs/Cellsam-nature.pdf`)：

1. 训练构建使用 **10 个数据来源**（不是 15）  
2. `LIVECell` 被明确作为 zero-shot/few-shot 的 held-out 数据  
3. 图像统一 upsample 到 `1,024 x 1,024` 作为 CellSAM 输入  
4. NeurIPS challenge 口径:  
   - NeurIPS training set = 标准数据集的 train/val/test + NeurIPS train/tuning  
   - open test 用于验证，hidden test 用于最终报告

可复核页码:
- 两阶段与 neck 叙述: `docs/Cellsam-nature.pdf` 第 3 页
- 数据集构建与 NeurIPS 设置: `docs/Cellsam-nature.pdf` 第 10 页

> 修订说明: 本文档 2026-03-04 版本中的“15 数据集 / 124 数据集评估”口径不适用于当前 Nature 2025 版本，已更正为 Methods 可复核口径。

### 2.2 两阶段训练细节 (Methods)

论文描述为两阶段:

1. Stage 1 (检测侧):
- 将 GT masks 转为 GT boxes
- 联合训练 ViT backbone + CellFinder

2. Stage 2 (分割侧对齐):
- 冻结 SAM-ViT
- 训练 neck，使 CellFinder 改变后的 ViT 特征重新对齐分割分支

Methods 超参条目:

- CellFinder 训练:
  - lr: head `1e-4`, SAM-ViT backbone `1e-5`
  - wd `1e-4`
  - clip norm `0.1`
  - AdamW + step scheduler（1960 epoch 后降 10 倍）
  - 2800 epochs, batch size 4, 8x H100

- Stage 2 neck fine-tune:
  - 训练目标: GT boxes -> individual-cell segmentation masks
  - lr `1e-4`, wd `1e-4`
  - AdamW
  - 不做 gradient clipping
  - 另一个 Methods 段落明确给出: 50 epochs + cosine learning rate schedule

可复核页码:
- CellFinder/neck 超参: `docs/Cellsam-nature.pdf` 第 10 页
- 50 epoch cosine 描述: `docs/Cellsam-nature.pdf` 第 11 页

### 2.3 官方指标 (按任务阶段拆分)

1. CellFinder 检测侧 (开发指标):
- COCO metrics（论文写明用于 CellFinder development）
- 重点报告 `mAP` 与 `AP50`
- IoU 阈值 0.5 到 0.95，步长 0.05
- 由于细胞密集，max detections 从 100 调到 10,000

2. 分割侧与主文 benchmark:
- 论文主对比指标是 `F1 error (1 - F1)`（对比 Cellpose）
- human vs model 分析中，明确使用 Recall / Precision / F1 公式

结论:
- 论文主文不是以 PQ/AJI 为主指标。
- “识别(检测) 与分割”分别有指标口径：检测侧偏 COCO，分割侧偏 F1 系列。

可复核页码:
- F1 error 主指标: 第 3-4 页
- Recall/Precision/F1 + COCO/mAP/AP50: 第 11 页

### 2.4 证据边界: loss 该怎么写

1. 论文 Methods 给出两阶段训练策略和关键超参，但公开仓库不含完整 Stage-2 训练脚本。  
2. 因此不能把 Stage-2 loss 细节写成“代码级已完全复现”。  
3. `sam_inference.py` 是推理路径，不包含 Dice/BCE 的 Stage-2 训练实现。  
4. 检测侧可代码级确认的损失来自 AnchorDETR `SetCriterion`：
- 分类: focal (`loss_ce`)
- 框回归: L1 (`loss_bbox`)
- 框几何: GIoU (`loss_giou`)

代码定位:
- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:140`
- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:191`
- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:223`
- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:231`
- `cellSAM_source/cellSAM/sam_inference.py`

### 2.5 CellSAM 的通道定义 (官方推理口径)

CellSAM 对 multiplex 图像给出的通道顺序:

- `(blank, nuclear, whole-cell)`

whole-cell 可选；缺失时走核分割路径。

代码定位:
- `cellSAM_source/cellSAM/cellsam_pipeline.py:85`
- `cellSAM_source/cellSAM/cellsam_pipeline.py:88`
- `cellSAM_source/cellSAM/cellsam_pipeline.py:158`

---

## 3. A2 结论复验 (2026-03-05)

### 3.1 CellFinder backbone vs model_cp encoder

复验脚本:
- `tools/_audit_cellfinder_backbone_compare.py`

复验结果:

1. `cellfinder_backbone` vs `model.image_encoder(no neck)`:
- same=171, diff=0

2. `cellfinder_backbone` vs `model_cp.image_encoder(no neck)`:
- same=0, diff=171

3. `model.image_encoder(no neck)` vs `model_cp.image_encoder(no neck)`:
- same=0, diff=171

结论:
- `cellfinder backbone` 对齐的是 `model` 分支的 ViT 主体，不是 `model_cp`。  
- 截图中的“73 same / 98 diff 且 model 与 model_cp encoder 完全一致”与本地复验不一致。

### 3.2 为什么会出现“model 与 model_cp 看起来一致”的误判

`CellSAM.load_state_dict()` 存在 fallback:

- 如果 checkpoint 中没有 `model_cp.*`，会把 `model` 复制给 `model_cp`

代码定位:
- `cellSAM_source/cellSAM/sam_inference.py:402`
- `cellSAM_source/cellSAM/sam_inference.py:406`

若使用了这类 checkpoint 或错误加载流程，就可能得到“model == model_cp”的假象。

---

## 4. 与我们项目数据的对应关系

### 4.1 我们的 Allen 数据 (项目训练数据)

本项目主数据仍是 Allen 心肌细胞:

- 总图像: 478
- 固定划分: Train 334 / Val 71 / Test 73
- 输入通道来自 BF/DAPI/Actn2 映射，不是 CellSAM 原论文混合集

### 4.2 写作口径建议

论文中保持两层口径:

1. CellSAM 原论文: 10 数据来源 + LIVECell held-out + NeurIPS challenge 规则
2. 本项目: Allen 单域微调与评估

不要写成“我们在 NeurIPS challenge 上训练/评估”，除非确实运行了该实验。

---

## 5. 后续补全建议 (论文前)

1. 从 Supplementary Table 1 抽取 10 数据来源的通道类型与样本统计，落地为附录表。  
2. `paper_preparation.md` 里的 Stage2/loss 表述统一引用本文件，避免过强结论。  
3. 若要写“官方识别与分割阶段指标”，统一写为:
- 检测侧: COCO mAP/AP50（CellFinder development）
- 分割侧: F1 error(1-F1) + Recall/Precision/F1（benchmark/human comparison）
