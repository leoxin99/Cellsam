# CellSAM 官方方案 vs 我们项目方案对照 (Codex 整理, 2026-02-28)

> 作者: **Codex (A1)**  
> 目的: 明确哪些部分沿用了官方 CellSAM, 哪些部分是本项目自定义设计, 并给出代码定位。  
> 口径说明:  
> 1. “官方”优先指 `cellSAM_source/` 中公开推理代码。  
> 2. “官方训练流程”主要来自 CellSAM 论文描述; 公开仓库未提供完整 Stage 2 训练脚本。  
> 3. “我们项目”指 `src/` + `tools/` 当前主线实现。  

---

## 1. 一句话结论

当前项目不是“直接复现官方 CellSAM pipeline”，而是：

1. **保留了 CellSAM/SAM 的分割模型底座**
2. **替换了官方检测器 CellFinder**
3. **重写了训练流程、推理流程、损失函数、评估与多通道输入策略**

因此更准确的描述是：

> **本项目以 CellSAM 为 foundation model，但检测、训练、推理和评估均做了面向心肌细胞任务的自定义改造。**

---

## 2. 仍然沿用官方 CellSAM 的部分

### 2.1 模型实例化

仍然通过官方 `get_model()` 构造模型：

- `cellSAM_source/cellSAM/model.py:43`
- `src/train.py:139`
- `src/inference/core.py:84`

### 2.2 SAM 主体结构

仍然沿用官方 `CellSAM` 内部的 SAM 组件：

1. `image_encoder`
2. `prompt_encoder`
3. `mask_decoder`

官方结构定义：

- `cellSAM_source/cellSAM/sam_inference.py:123`

本项目训练/推理调用：

- `src/train.py:269`
- `src/train.py:328`
- `src/train.py:331`
- `src/inference/core.py:178`
- `src/inference/core.py:187`
- `src/inference/core.py:194`

### 2.3 CellSAM 双分支对象

官方 `CellSAM` 内部有两套分支：

1. `self.model`
2. `self.model_cp`

代码：

- `cellSAM_source/cellSAM/sam_inference.py:136`
- `cellSAM_source/cellSAM/sam_inference.py:137`

这不是本项目新发明的结构，是官方对象本身就有。

---

## 3. 本项目没有采用官方结构、而是自己设计的部分

## 3.1 检测器: 官方 CellFinder 被完全替换

### 官方

官方检测是 `CellfinderAnchorDetr` / `CellFinder`：

- `cellSAM_source/cellSAM/sam_inference.py:62`
- `cellSAM_source/cellSAM/sam_inference.py:134`
- `cellSAM_source/cellSAM/sam_inference.py:234`

### 我们

本项目主线检测器改成了：

1. DAPI 核检测
2. 双核合并
3. 各向异性框扩展
4. Actn2 过滤
5. Z-line 自适应框

代码：

- `src/detection/dapi.py:32`
- `src/detection/dapi.py:71`
- `src/detection/dapi.py:150`
- `src/detection/dapi.py:231`
- `src/detection/dapi.py:274`
- `src/detection/dapi.py:337`
- `src/detection/dapi.py:461`
- `src/detection/dapi.py:539`

### 结论

这是本项目**偏离官方最大**的一处。  
原因是心肌细胞上官方 CellFinder 效果不佳，所以项目转向了可控的 DAPI / Adaptive 检测路线。

---

## 3.2 推理入口: 没走官方 `predict()` / `segment_cellular_image()`

### 官方

官方高层推理入口：

1. `segment_cellular_image(...)`
2. 其内部调用 `model.predict(...)`

代码：

- `cellSAM_source/cellSAM/model.py:114`
- `cellSAM_source/cellSAM/model.py:165`

### 我们

本项目改成了统一推理核心：

- `src/inference/core.py:146` `segment_with_boxes(...)`

并把验证、Oracle 评估、E2E 评估统一到这个入口。

### 结论

官方是“高层封装推理”，我们是“拆开后重写统一推理内核”。

---

## 3.3 官方 `model_cp` 分支没有被原样沿用

### 官方

官方 `predict()` 在 `adv_mode=True` 时，会优先使用 `model_cp`：

- 编码器: `cellSAM_source/cellSAM/sam_inference.py:208`
- 分割分支选择: `cellSAM_source/cellSAM/sam_inference.py:327`

即：

```python
mdl = self.model_cp if self.adv_mode else self.model
```

### 我们

项目历史上大量训练/推理代码直接调用 `model.model.*`：

- `src/train.py:269`
- `src/train.py:328`
- `src/train.py:331`
- `src/inference/core.py:178`
- `src/inference/core.py:187`
- `src/inference/core.py:194`

后续为了兼容官方权重，又加了：

- `src/train.py:147`

```python
model.model.load_state_dict(model.model_cp.state_dict())
```

### 结论

现在主线不是“按官方逻辑切换 `model_cp`”，而是：

1. 先把 `model_cp` 权重复制到 `model.model`
2. 再继续走本项目自定义训练/推理代码

这是一种**兼容性修复方案**，不是官方原生结构。

---

## 3.4 数据输入与预处理: 官方和我们差异很大

### 官方预处理

官方 `segment_cellular_image()` 的高层流程：

1. `format_image_shape(img)` 保证输入是 3 通道
2. 可选 `normalize_image(img)`  
3. 转为 `CHW` tensor
4. 调 `model.predict(...)`

代码：

- `cellSAM_source/cellSAM/model.py:147`
- `cellSAM_source/cellSAM/model.py:158`
- `cellSAM_source/cellSAM/model.py:165`

更细的官方预测预处理在 `predict()` 内部：

1. `prep_2(images, percentile=True)`  
2. `Resize((1024, 1024))`
3. `sam_preprocess_pad(...)` padding
4. `PercentileThreshold()`
5. `self.normalize(...)` (ImageNet mean/std)
6. `anchorT.Standardize()`
7. `forward(...)`
8. `sam_preprocess(..., div_255=True)`

代码：

- `cellSAM_source/cellSAM/sam_inference.py:220`
- `cellSAM_source/cellSAM/sam_inference.py:224`
- `cellSAM_source/cellSAM/sam_inference.py:226`
- `cellSAM_source/cellSAM/sam_inference.py:228`
- `cellSAM_source/cellSAM/sam_inference.py:190`

### 我们的预处理

训练数据集完全是自定义逻辑：

1. 固定 train/val/test split
2. Albumentations 增强
3. 每通道归一化，或 BF-only 复制三通道
4. 可选语义通道映射 `R=BF, G=Actn2, B=DAPI`
5. 可选 2ch 模式 `B=BF(copy)`
6. 输出 float `[0,1]`

代码：

- `src/augmented_dataset.py:25`
- `src/augmented_dataset.py:191`
- `src/augmented_dataset.py:268`
- `src/augmented_dataset.py:320`
- `src/augmented_dataset.py:384`
- `src/augmented_dataset.py:388`

训练/推理前进一步送入：

- `model.sam_preprocess(images)`  

代码：

- `src/train.py:265`
- `src/inference/core.py:178`

### 关键差异总结

官方有：

1. `PercentileThreshold`
2. `ImageNet normalize`
3. `Standardize`
4. `ToRGB` (bbox branch)
5. 官方 `predict()` 内部的 `model_cp` 路径

我们没有完整沿用这些，而是：

1. 自己做数据归一化/复制/语义映射
2. 直接喂给 `model.sam_preprocess`
3. 走自定义 `segment_with_boxes`

---

## 3.5 训练流程: 官方 Stage1/Stage2 与我们完全不同

## 3.5.1 官方训练流程 (论文口径)

### Stage 1

论文描述：Stage 1 重点训练检测器 CellFinder，并联合共享 backbone 学习细胞框。

对应结构证据：

- CellFinder 模块在官方模型中存在: `cellSAM_source/cellSAM/sam_inference.py:134`

但**公开仓库没有完整 Stage 1 训练脚本**。

### Stage 2

论文描述：Stage 2 做分割侧对齐，核心思想是冻结大部分 SAM 主干，只训练少量对齐模块（通常被理解为 neck 侧对齐）。

但**公开仓库同样没有完整 Stage 2 训练脚本**。  
因此对 Stage 2 的精确 loss、优化器细节、每一步调度，不能从公开代码逐行复现，只能依赖论文描述。

### 结论

官方训练流程是：

1. **两阶段**
2. **先检测，再分割对齐**
3. **公开仓库以推理代码为主，训练链路不完整**

## 3.5.2 我们的训练流程 (代码可证)

本项目当前训练主入口：

- `src/train.py`

核心流程如下。

### 第 1 步: 构建固定数据划分

- `src/train.py:67`
- `src/train.py:72`

使用固定 `train_ids` / `val_ids`，构建 `AugmentedAllenDataset`。

### 第 2 步: 从 GT instance mask 自动生成 GT boxes

在数据集里把每个实例 mask 转为 box：

- `src/augmented_dataset.py:320`

并可在训练时做 box perturbation：

- `src/augmented_dataset.py:268`
- `src/augmented_dataset.py:349`

### 第 3 步: 创建模型并冻结部分参数

- `src/train.py:128`

主线逻辑：

1. `get_model()`
2. `model_cp -> model.model` 权重拷贝
3. 冻结 `image_encoder`
4. 可选冻结 `mask_decoder`
5. 可选插入 LoRA
6. 可选加入 adapter

代码：

- `src/train.py:147`
- `src/train.py:156`
- `src/train.py:161`
- `src/train.py:171`

### 第 4 步: 逐图编码、逐 box 解码

训练时先对整张图编码一次：

- `src/train.py:269`

然后遍历同一图中每个 GT box：

- `src/train.py:322`

每个 box 都调用：

1. `prompt_encoder`
2. `mask_decoder`
3. 上采样到 `1024x1024`

代码：

- `src/train.py:328`
- `src/train.py:331`
- `src/train.py:339`

### 第 5 步: 目标是“当前 box 对应的单个 cell”

训练目标不是整图 semantic mask，而是：

```python
target = (sample_mask == cell_id).float()
```

代码：

- `src/train.py:348`
- `src/train.py:392`

### 第 6 步: 在 box 区域内裁剪 loss

预测和目标都只在 box 扩展区域内参与 loss：

- `src/train.py:357`
- `src/train.py:401`

### 第 7 步: 自定义 `CombinedLoss`

损失由本项目自己定义：

- `src/losses/combined.py:439`

包含 Dice/BCE + 多种结构项。

### 第 8 步: 验证也走自定义统一推理核心

不是用官方 `predict()` 验证，而是：

- `src/train.py:465`

内部调用：

- `src/inference/core.py:146`

### 我们与官方训练的本质差异

| 项 | 官方论文流程 | 我们当前流程 |
|---|---|---|
| 检测训练 | Stage1 训练 CellFinder | 不训练 CellFinder |
| 分割训练 | Stage2 对齐 | GT box instance-level fine-tuning |
| 训练对象 | 官方两阶段 | 当前主线训练 prompt encoder + mask decoder（encoder 冻结） |
| loss | 论文描述，公开代码不完整 | `CombinedLoss` 全自定义 |
| 验证 | 官方未公开完整脚本 | `segment_with_boxes()` + 统一 metrics |

---

## 3.6 损失函数体系几乎完全是项目自定义

本项目在 `src/losses/combined.py` 里自定义实现了：

1. `BoundaryLoss`
2. `AJILoss`
3. `TopologyLoss`
4. `SizeLoss`
5. `ContourLoss`
6. `NeighborIntrusionLoss`
7. `OverlapMutexLoss`
8. `CombinedLoss`

代码：

- `src/losses/combined.py:167`
- `src/losses/combined.py:225`
- `src/losses/combined.py:290`
- `src/losses/combined.py:381`
- `src/losses/combined.py:414`
- `src/losses/combined.py:439`

此外还有：

1. 动态 `pos_weight`
   - `src/losses/combined.py:546`
2. computability gating
   - `src/losses/combined.py:580`
   - `src/losses/combined.py:585`

这整套都不是官方 CellSAM 公开代码的一部分。

---

## 3.7 冲突像素裁决是自定义的，不是官方原始规则

本项目统一推理核心定义了三种冲突策略：

1. `argmax_prob`
2. `first_write`
3. `last_write`

代码：

- `src/inference/core.py:28`
- `src/inference/core.py:275`
- `src/inference/core.py:301`

官方 `predict()` 的实例 mask 生成逻辑更接近：

1. 每个 mask 编号
2. 逐像素取最大编号

即官方并没有像本项目一样把冲突策略显式抽象成统一配置。

---

## 3.8 多通道语义映射、adapter、LoRA 都是项目新增

### 语义通道映射

- `src/augmented_dataset.py:25`

定义：

1. `R <- BF`
2. `G <- Actn2`
3. `B <- DAPI`

### Channel Adapter

- `src/adapters/channel_adapter.py:21`
- `src/adapters/channel_adapter.py:108`

### LoRA

- `src/lora.py:49`
- `src/lora.py:87`

官方 CellSAM 没有这些模块。

---

## 4. 官方后处理函数是什么

官方 `segment_cellular_image()` 在 `predict()` 返回后，会执行下面几类后处理。

## 4.1 `fill_holes_and_remove_small_masks`

入口：

- `cellSAM_source/cellSAM/model.py:175`

实现：

- `cellSAM_source/cellSAM/utils.py:240`

作用：

1. 对每个实例填洞 (`binary_fill_holes`)
2. 删除面积小于 `min_size` 的小碎片实例
3. 重新连续编号

这一步的目标是：

> 清理小噪声实例、修复 mask 内部空洞，让输出实例 mask 更规整。

## 4.2 `postprocess_predictions`

入口：

- `cellSAM_source/cellSAM/model.py:172`

实现：

- `cellSAM_source/cellSAM/model.py:182`

作用：

对每个实例依次做：

1. 去 holes / 去 islands
2. opening / closing
3. dilation + erosion
4. Gaussian smoothing
5. 再阈值化
6. 最后合并成整张实例 mask

这是一套**比较强的形态学清洗**，适用于噪声图像，但会改变实例边界。

## 4.3 `remove_boundaries`

入口：

- `cellSAM_source/cellSAM/model.py:176`

实现：

- `cellSAM_source/cellSAM/utils.py:122`
- `cellSAM_source/cellSAM/utils.py:125`

作用：

1. 先求实例边界 `_mask_outline`
2. 再从 mask 中减去边界像素

也就是：

> 把每个细胞实例外圈 1 像素边界去掉，减少 touching instances 的边界粘连。

---

## 5. 官方 WSI / tile / cell-size gaging 结构是什么

这部分官方高层入口在：

- `cellSAM_source/cellSAM/cellsam_pipeline.py:54`

## 5.1 `use_wsi`

参数位置：

- `cellSAM_source/cellSAM/cellsam_pipeline.py:61`

含义：

> 是否使用大图 / whole-slide inference 的 tiled 方式分块推理。

若 `use_wsi=True`，则调用：

- `segment_wsi(...)`
- `cellSAM_source/cellSAM/cellsam_pipeline.py:199`

适合大图或高密度图像，不适合你们这种固定 1024 patch 小图主线。

## 5.2 tile / block / overlap

相关参数：

- `block_size`
- `overlap`
- `iou_depth`
- `iou_threshold`

位置：

- `cellSAM_source/cellSAM/cellsam_pipeline.py:19`
- `cellSAM_source/cellSAM/cellsam_pipeline.py:23`
- `cellSAM_source/cellSAM/cellsam_pipeline.py:118`

含义：

1. `block_size`: tile 大小
2. `overlap`: tile 之间重叠区域
3. `iou_depth`: 拼接时用于合并实例的深度范围参数
4. `iou_threshold`: 跨 tile label merge 的 IoU 阈值

本质上是：

> 先分块做局部分割，再在重叠区域里合并实例标签。

## 5.3 `gauge_cell_size`

参数位置：

- `cellSAM_source/cellSAM/cellsam_pipeline.py:62`

实现入口：

- `cellSAM_source/cellSAM/cellsam_pipeline.py:10`

逻辑：

1. 先跑一轮分割
2. 统计当前图里的细胞中位尺寸 `get_median_size`
3. 再根据尺寸判断是否要切换到 WSI 模式 / 调整参数

代码链：

- `cellSAM_source/cellSAM/cellsam_pipeline.py:10`
- `cellSAM_source/cellSAM/utils.py:88`

作用可以理解为：

> 先试跑一遍，粗估当前图里的细胞大小，再决定是否要用 tiled WSI 方案。

---

## 6. `model_cp` 到底是什么

## 6.1 它不是 Cellpose

`model_cp` 里的 `cp` **不是 Cellpose**。  
它定义在官方 CellSAM 类里：

- `cellSAM_source/cellSAM/sam_inference.py:137`

```python
self.model_cp = copy.deepcopy(self.model)
```

也就是说：

1. 先创建一个 SAM `self.model`
2. 再深拷贝一份，得到 `self.model_cp`

这是 **CellSAM 内部的第二套 SAM 分支**，不是 Cellpose 模型。

## 6.2 它是 CellSAM 官方权重体系中的“adv_mode 分支”

在官方 `forward()` 与 `predict()` 中，只要 `adv_mode=True`：

- 编码走 `model_cp.image_encoder`
  - `cellSAM_source/cellSAM/sam_inference.py:208`
- prompt/mask 解码走 `model_cp`
  - `cellSAM_source/cellSAM/sam_inference.py:327`

并且在 `load_state_dict()` 里，如果 checkpoint 中存在 `model_cp.*` key，那么就维持 `adv_mode=True`：

- `cellSAM_source/cellSAM/sam_inference.py:401`
- `cellSAM_source/cellSAM/sam_inference.py:405`

## 6.3 当前最严谨的解释

`model_cp` 应理解为：

> **CellSAM 官方 checkpoint 中用于 adv_mode 推理的那套 SAM 权重分支。**

它属于 CellSAM 官方模型内部结构的一部分，不是 Cellpose，也不是外部附加模型。

---

## 7. 当前项目与官方的核心差异汇总

| 模块 | 官方 CellSAM | 我们项目 |
|---|---|---|
| 检测 | CellFinder / Anchor-DETR | DAPI / Adaptive |
| 推理入口 | `segment_cellular_image()` + `predict()` | `segment_with_boxes()` |
| 权重分支 | `adv_mode -> model_cp` | 历史上直接用 `model.model`，后加 copy 修复 |
| 预处理 | PercentileThreshold + Normalize + Standardize + 官方 predict 链 | 数据集自定义归一化 + BF复制/语义映射 + `sam_preprocess` |
| 训练流程 | 论文两阶段，公开训练代码不完整 | `src/train.py` 自定义 instance-level 训练 |
| 训练对象 | 论文 Stage1/2 | 当前主线: encoder 冻结，训练 prompt encoder + mask decoder |
| loss | 论文有描述，公开实现不完整 | `CombinedLoss` 自定义 |
| 后处理 | fill holes / remove small / optional morphology / optional remove boundaries | 统一 `InferenceConfig` + clipping + conflict policy + optional size validation |
| 多通道 | 官方支持 RGB / multiplex 输入 | 语义通道映射 + adapter |
| LoRA | 官方无 | 本项目新增 |

---

## 8. 当前阅读建议

如果你想继续顺着代码理解，建议按这个顺序看：

1. 官方模型骨架  
   - `cellSAM_source/cellSAM/sam_inference.py`
2. 官方高层推理封装  
   - `cellSAM_source/cellSAM/model.py`
   - `cellSAM_source/cellSAM/cellsam_pipeline.py`
3. 我们的训练入口  
   - `src/train.py`
4. 我们的统一推理入口  
   - `src/inference/core.py`
5. 我们的检测器  
   - `src/detection/dapi.py`
6. 我们的损失  
   - `src/losses/combined.py`

---

## 9. 官方预处理 vs 我们项目预处理: 表格对照

> 说明: 本节区分三种口径  
> 1. **官方高层入口**: `segment_cellular_image()`  
> 2. **官方低层核心**: `prep_2() + forward()`  
> 3. **我们项目主线**: `src/train.py` + `src/inference/core.py`

| 维度 | 官方高层 `segment_cellular_image()` | 官方低层 `prep_2() + forward()` | 我们项目旧主线 | 我们项目当前主线 |
|---|---|---|---|---|
| 输入来源 | `numpy` 图像 | tensor list | `AugmentedAllenDataset` 输出 | `AugmentedAllenDataset` 输出 |
| 输入范围 | 原始图像 | 期望经转换后进入 `[0,255]` / `[0,1]` 混合链路 | dataset 已归一化到 `[0,1]` | dataset 已归一化到 `[0,1]` |
| shape 整理 | `format_image_shape()` | `Resize((1024,1024))` + `sam_preprocess_pad()` | 数据集已对齐尺寸 | 数据集已对齐尺寸 |
| 百分位裁剪 | `normalize_image()` 内部 percentile | `PercentileThreshold()` | 无 | 通过 `model.prep_2(..., percentile=True)` 使用官方实现 |
| ImageNet 标准化 | 无单独显式层 | `self.normalize(...)` | 无 | 通过 `model.prep_2(...)` 使用官方实现 |
| min-max 标准化 | 无单独显式层 | `anchorT.Standardize()` | 无 | 通过 `model.prep_2(...)` 使用官方实现 |
| SAM 标准化 | `predict()` 内部走 `sam_preprocess(div_255=True)` | `sam_preprocess(div_255=True)` | `sam_preprocess(div_255=False)` | `official_preprocess_*()` 内部走 `sam_preprocess(div_255=True)` |
| 编码器分支 | `model_cp` (`adv_mode=True`) | `model_cp` (`adv_mode=True`) | 历史上常是 `model.model` | 当前是 `model.model_cp` |
| 代码位置 | `cellSAM_source/cellSAM/model.py:114` | `cellSAM_source/cellSAM/sam_inference.py:217`, `cellSAM_source/cellSAM/sam_inference.py:201` | 历史见 `docs/cellsam_update_predict_2.28.md:74`, `docs/cellsam_update_predict_2.28.md:258` | `src/official_preprocess.py:24`, `src/official_preprocess.py:59`, `src/train.py:261`, `src/inference/core.py:188` |

### 9.1 官方预处理流程图

```text
原始图像
  -> format_image_shape()
  -> normalize_image()              [高层入口可选]
  -> predict()
      -> prep_2()
          -> Resize(1024)
          -> sam_preprocess_pad()   [只做 padding]
          -> PercentileThreshold()
          -> ImageNet Normalize
          -> Standardize
      -> forward()
          -> sam_preprocess(div_255=True)
          -> model_cp.image_encoder()
```

对应代码:

- `cellSAM_source/cellSAM/model.py:147`
- `cellSAM_source/cellSAM/model.py:158`
- `cellSAM_source/cellSAM/model.py:165`
- `cellSAM_source/cellSAM/sam_inference.py:217`
- `cellSAM_source/cellSAM/sam_inference.py:201`

### 9.2 我们项目旧主线流程图

```text
AugmentedAllenDataset 输出 [0,1]
  -> model.sam_preprocess(div_255=False)
  -> model.model.image_encoder()
  -> prompt_encoder / mask_decoder
```

对应历史证据:

- `docs/cellsam_update_predict_2.28.md:74`
- `docs/cellsam_update_predict_2.28.md:200`
- `docs/cellsam_update_predict_2.28.md:258`

### 9.3 我们项目当前主线流程图

```text
AugmentedAllenDataset 输出 [0,1]
  -> official_preprocess_only() / official_preprocess_and_encode()
      -> *255
      -> model.prep_2(percentile=True)
      -> model.forward()
          -> sam_preprocess(div_255=True)
          -> model_cp.image_encoder()
  -> model_cp.prompt_encoder()
  -> model_cp.mask_decoder()
```

对应代码:

- `src/official_preprocess.py:24`
- `src/official_preprocess.py:59`
- `src/train.py:261`
- `src/train.py:276`
- `src/inference/core.py:188`

---

## 10. CellSAM 官方完整推理结构

本节只描述 `cellSAM_source/` 中**公开可核验**的推理链路，不延伸推测未公开训练实现。

### 10.1 入口分层

官方推理有三层入口:

1. **应用层入口**: `cellsam_pipeline(...)`
   - `cellSAM_source/cellSAM/cellsam_pipeline.py:54`
2. **单图入口**: `segment_cellular_image(...)`
   - `cellSAM_source/cellSAM/model.py:114`
3. **模型级入口**: `CellSAM.predict(...)`
   - `cellSAM_source/cellSAM/sam_inference.py:286`

它们的关系是:

```text
cellsam_pipeline()
  -> segment_cellular_image()
      -> CellSAM.predict()
```

### 10.2 `cellsam_pipeline()` 做什么

`cellsam_pipeline()` 是官方最高层封装，负责:

1. 加载模型
2. 图像归一化/增强
3. 决定走单图还是 WSI/tile 路线
4. 可选根据细胞大小自动调推理模式

相关代码:

- `cellSAM_source/cellSAM/cellsam_pipeline.py:54`
- `cellSAM_source/cellSAM/cellsam_pipeline.py:61`
- `cellSAM_source/cellSAM/cellsam_pipeline.py:62`
- `cellSAM_source/cellSAM/cellsam_pipeline.py:199`
- `cellSAM_source/cellSAM/cellsam_pipeline.py:202`

### 10.3 `segment_cellular_image()` 做什么

`segment_cellular_image()` 是最常用的单图入口，负责:

1. `format_image_shape()` 把输入整理成标准 3 通道格式
2. 可选 `normalize_image()`
3. 调 `model.predict(...)`
4. 在 `predict()` 返回后执行:
   - 可选 `postprocess_predictions()`
   - `fill_holes_and_remove_small_masks()`
   - 可选 `subtract_boundaries()`

相关代码:

- `cellSAM_source/cellSAM/model.py:147`
- `cellSAM_source/cellSAM/model.py:158`
- `cellSAM_source/cellSAM/model.py:165`
- `cellSAM_source/cellSAM/model.py:172`
- `cellSAM_source/cellSAM/model.py:175`
- `cellSAM_source/cellSAM/model.py:176`

### 10.4 `predict()` 的内部流程

`CellSAM.predict()` 是真正做 box-prompt 分割的核心。

#### Step 1: 生成图像 embedding

若用户传的是整图，则先走:

1. `generate_embeddings()`
2. 内部调用 `prep_2(images, percentile=True)`
3. 再调用 `forward()`

代码:

- `cellSAM_source/cellSAM/sam_inference.py:258`
- `cellSAM_source/cellSAM/sam_inference.py:217`
- `cellSAM_source/cellSAM/sam_inference.py:201`

#### Step 2: 没有 box 时先用 CellFinder 产框

如果 `predict()` 没收到 `bounding_boxes`，则调用:

1. `generate_bounding_boxes()`
2. 其内部走 `sam_bbox_preprocessing()`
3. 再调 `self.cellfinder.forward_inference(...)`
4. 动态阈值筛出框

代码:

- `cellSAM_source/cellSAM/sam_inference.py:234`
- `cellSAM_source/cellSAM/sam_inference.py:168`
- `cellSAM_source/cellSAM/sam_inference.py:134`

#### Step 3: 逐框做 prompt segmentation

对每个 box:

1. 选择分支: `mdl = self.model_cp if self.adv_mode else self.model`
2. `prompt_encoder(boxes=...)`
3. `mask_decoder(...)`
4. `postprocess_masks()` 上采样回原图
5. 用 `iou_predictions` 做阈值过滤
6. 用 `mask_threshold` 做二值化

代码:

- `cellSAM_source/cellSAM/sam_inference.py:327`
- `cellSAM_source/cellSAM/sam_inference.py:333`
- `cellSAM_source/cellSAM/sam_inference.py:339`
- `cellSAM_source/cellSAM/sam_inference.py:350`
- `cellSAM_source/cellSAM/sam_inference.py:354`
- `cellSAM_source/cellSAM/sam_inference.py:359`

#### Step 4: 多实例 mask 合并

官方将每个二值 mask 乘上实例 id，再做 `np.max` 合并:

- `cellSAM_source/cellSAM/sam_inference.py:388`
- `cellSAM_source/cellSAM/sam_inference.py:391`

这等价于:

> 如果两个实例在同一像素都为前景，编号更大的实例覆盖编号更小的实例。

### 10.5 官方推理总流程图

```text
cellsam_pipeline()
  -> segment_cellular_image()
      -> format_image_shape()
      -> normalize_image()                          [optional]
      -> CellSAM.predict()
          -> generate_embeddings()
              -> prep_2()
                  -> Resize(1024)
                  -> sam_preprocess_pad()
                  -> PercentileThreshold()
                  -> ImageNet Normalize
                  -> Standardize
              -> forward()
                  -> sam_preprocess(div_255=True)
                  -> model_cp.image_encoder()
          -> if no boxes:
              -> generate_bounding_boxes()
                  -> sam_bbox_preprocessing()
                  -> cellfinder.forward_inference()
          -> for each box:
              -> model_cp.prompt_encoder()
              -> model_cp.mask_decoder()
              -> IoU filtering
              -> mask thresholding
          -> np.max merge
      -> postprocess_predictions()                  [optional]
      -> fill_holes_and_remove_small_masks()
      -> subtract_boundaries()                      [optional]
```

### 10.6 与我们项目推理主线的直接差异

当前项目主线 `src/inference/core.py` 的差异点:

1. 只接受外部给定 boxes，不用官方 CellFinder
2. 冲突像素裁决改成 `argmax_prob / first_write / last_write`
3. 默认 `apply_postprocess=False`
4. 增加了 `box_clipping`
5. 现在编码端已切到 `official_preprocess_and_encode()` + `model_cp`

代码:

- `src/inference/core.py:18`
- `src/inference/core.py:146`
- `src/inference/core.py:188`
- `src/inference/core.py:222`
- `src/inference/core.py:275`

### 10.7 图中模块 -> 代码文件 -> 作用 -> 输入输出

| 图中模块 | 代码文件 | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| Vision transformer / image encoder | `cellSAM_source/cellSAM/sam_inference.py:127`, `C:\Users\71069\.conda\envs\cellsam\lib\site-packages\segment_anything\modeling\image_encoder.py:17` | SAM ViT-B 主干, 提取图像特征并经 neck 压到 256 通道 | `B x 3 x 1024 x 1024` 图像 tensor | `B x 256 x 64 x 64` image embedding |
| ViT 主体 (patch + blocks) | `C:\Users\71069\.conda\envs\cellsam\lib\site-packages\segment_anything\modeling\image_encoder.py:58`, `image_encoder.py:72`, `image_encoder.py:119` | patchify + 12 层 transformer block | `B x 3 x 1024 x 1024` | `B x 64 x 64 x 768` token feature |
| neck / downsample | `C:\Users\71069\.conda\envs\cellsam\lib\site-packages\segment_anything\modeling\image_encoder.py:88` | 把 768 通道 token feature 投影到 256 通道, 供 SAM decoder 使用 | `B x 768 x 64 x 64` | `B x 256 x 64 x 64` |
| Prompt encoder | `cellSAM_source/cellSAM/sam_inference.py:333`, `C:\Users\71069\.conda\envs\cellsam\lib\site-packages\segment_anything\modeling\prompt_encoder.py:16` | 把 box / point / mask prompt 编成 sparse + dense embeddings | box/point/mask prompt | sparse prompt embeddings, dense prompt embeddings |
| Mask decoder | `cellSAM_source/cellSAM/sam_inference.py:339`, `C:\Users\71069\.conda\envs\cellsam\lib\site-packages\segment_anything\modeling\mask_decoder.py:16` | 用 image embedding + prompt embedding 预测低分辨率 mask 与 IoU 质量 | image embedding + prompt embeddings + image PE | low-res masks, iou predictions |
| CellFinder backbone | `cellSAM_source/cellSAM/AnchorDETR/models/backbone.py:208`, `backbone.py:219` | CellFinder 的特征主干, 只保留 SAM encoder 的 ViT 主体, 不含 neck | `B x 3 x 1024 x 1024` | `B x 768 x 64 x 64` feature map |
| AnchorDETR transformer decoder | `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:33`, `anchor_detr.py:121`, `cellSAM_source/cellSAM/AnchorDETR/models/transformer.py:24` | 检测头, 输出候选框分类和坐标 | backbone features + padding mask | `pred_logits`, `pred_boxes` |
| bbox postprocess | `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:368` | 把归一化框转换回图像坐标并筛出分数 | `pred_logits`, `pred_boxes`, target sizes | 最终检测框列表 |

---

## 11. 官方论文指标口径 vs 我们项目指标口径

### 11.1 CellSAM 论文不是“不看分割质量”

不是。CellSAM 论文的核心目标仍然是**实例分割性能**，只是论文主文选用的主指标不是 PQ/AJI，而是 **F1 error (1 - F1)**。

论文证据:

- Nature Methods 主文摘要与 Fig. 2 说明:
  - “Segmentation performance ... compared using segmentation error (1-F1)”
  - “CellSAM achieves human-level accuracy for generalized cell segmentation”
- 来源:
  - `https://www.nature.com/articles/s41592-025-02879-w`
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC12695629/`

所以论文不是“不追求分割效果”，而是:

> **追求实例分割效果，但采用 F1 / 1-F1 作为主 benchmark 口径。**

### 11.2 官方公开代码里能直接看到的指标

公开评估代码 `paper_evaluation/` 里主要计算:

1. `f1`
2. `recall`
3. `precision`
4. `dice`
5. `aji`
6. `mean_ap`

代码:

- `cellSAM_source/paper_evaluation/cpm.py:34`
- `cellSAM_source/paper_evaluation/cpm.py:47`
- `cellSAM_source/paper_evaluation/cpm.py:86`
- `cellSAM_source/paper_evaluation/cpm.py:87`
- `cellSAM_source/paper_evaluation/cpm.py:88`

其中:

- `F1 = TP / (TP + 0.5 * (FP + FN))`
- `Recall = TP / (TP + FN)`
- `Dice = 2TP / (2TP + FP + FN)`

代码:

- `cellSAM_source/paper_evaluation/cpm.py:60`
- `cellSAM_source/paper_evaluation/cpm.py:62`
- `cellSAM_source/paper_evaluation/cpm.py:86`
- `cellSAM_source/paper_evaluation/cpm.py:87`
- `cellSAM_source/paper_evaluation/cpm.py:89`

### 11.3 论文主文为何常写 `1-F1`

因为论文在展示“误差”时，用的是:

```text
segmentation error = 1 - F1
```

这样:

- `F1` 越大越好
- `1-F1` 越小越好

这是一种论文展示口径，不代表论文不关心 mask 质量。

### 11.4 与我们项目指标的区别

| 维度 | CellSAM 论文主口径 | 我们项目主口径 |
|---|---|---|
| 主指标 | `F1` / `1-F1` | `BM-1to1 Dice`, `PQ`, `AJI` |
| 关注点 | 整体实例匹配成功率 | 既看匹配，也看分割质量与粘连问题 |
| 优点 | 简洁、跨数据集容易比较 | 诊断信息更完整 |
| 局限 | 不容易拆出“检出 vs 边界”问题 | 指标体系更复杂 |

因此:

> **CellSAM 论文是“追求分割效果”，只是选择了 F1/1-F1 作为论文主 benchmark；我们项目则更强调 instance segmentation 诊断，因此使用了 PQ/AJI/BM-Dice。**

---

## 12. `model` / `model_cp` / `cellfinder` 的真实对象关系

### 12.1 论文描述的“共享 backbone”是什么意思

论文描述是:

> ViT features first go to CellFinder, and also go to the neck + mask decoder branch.

这是**架构层 / 功能层共享**的表述。

论文来源:

- Nature Methods inference paragraph
- PMC Fig. 1 / architecture text

也就是说，论文表达的是:

> 两个下游模块都消费“同一类 ViT 特征表示”。

它**不等价于**:

> 发布代码里一定把检测和分割写成同一个 Python `image_encoder` 实例对象。

### 12.2 发布代码对象层面的真实结构

官方 `CellSAM` 初始化时有三块:

1. `self.model` — 一套完整 SAM
2. `self.model_cp` — `copy.deepcopy(self.model)` 得到的另一套完整 SAM
3. `self.cellfinder` — 单独的 Anchor-DETR 检测器

代码:

- `cellSAM_source/cellSAM/sam_inference.py:134`
- `cellSAM_source/cellSAM/sam_inference.py:137`

```python
self.cellfinder = CellfinderAnchorDetr(config)
self.model_cp = copy.deepcopy(self.model)
```

所以从**对象实例**看，并不是:

```text
cellfinder 直接引用 self.model.image_encoder
```

而是:

```text
self.model        -> 一套 SAM
self.model_cp     -> 另一套 SAM
self.cellfinder   -> 自己内部再包一套 SAMBackbone
```

### 12.3 `cellfinder` 的 `SAMBackbone` 从哪里来

`CellfinderAnchorDetr` 内部构建检测头时，会调用:

- `cellSAM_source/cellSAM/AnchorDETR/models/__init__.py:13`
- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:412`

最终进入:

- `cellSAM_source/cellSAM/AnchorDETR/models/backbone.py:219`

```python
backbone = sam_model_registry[sam_vit]()
backbone = backbone.image_encoder
backbone = ModifiedImageEncoderViT(backbone)
```

这说明 `cellfinder` 的 backbone 是:

1. 先新建一套原始 SAM ViT
2. 取其中的 `image_encoder`
3. 包装成 `ModifiedImageEncoderViT`

也就是:

> **对象层面它既不是直接引用 `self.model.image_encoder`，也不是直接引用 `self.model_cp.image_encoder`。**

### 12.4 `backbone` 是什么，它和 encoder 的关系

在这里 `backbone` 基本可以理解为:

> 检测器前面的特征提取主干网络

对 CellFinder 来说，这个 backbone 就是 **SAM ViT 的 image encoder 主体**。

但要注意，官方实现里用了 `ModifiedImageEncoderViT`:

- `cellSAM_source/cellSAM/AnchorDETR/models/backbone.py:180`

它只保留:

1. `patch_embed`
2. `blocks`
3. `pos_embed`

**不包含 neck**。

也就是说:

> 对检测分支来说，backbone = image encoder 的“ViT 主体部分”，不含 neck。

### 12.5 关键验证: `cellfinder` backbone 实际等于谁

我在本地 `cellsam` 环境中重新对比了权重，结果是:

1. `cellfinder.decode_head.backbone.body`
   与
   `model.image_encoder` **去掉 neck 后**
   完全相同:
   - `same = 171`
   - `diff = 0`

2. `cellfinder.decode_head.backbone.body`
   与
   `model_cp.image_encoder` **去掉 neck 后**
   完全不同:
   - `same = 0`
   - `diff = 171`

这说明发布 checkpoint 中:

> **CellFinder 的 backbone 权重 = `model.image_encoder` 的非-neck部分**

而不是 `model_cp`。

> 术语对齐:
> - 在 **官方 `CellSAM` 类内部**，这里写作 `self.model.image_encoder`
> - 在 **我们项目脚本调用层**，外层变量常命名为 `model`，因此等价写法常表现为 `model.model.image_encoder`
> - 两者指的是同一个官方分支，只是一个是类内写法，一个是类外访问写法

#### 本地实测小表

| 对比对象 | same | diff | 结论 |
|---|:---:|:---:|---|
| `cellfinder.decode_head.backbone.body` vs `model.image_encoder` 去 neck | 171 | 0 | **完全一致** |
| `cellfinder.decode_head.backbone.body` vs `model_cp.image_encoder` 去 neck | 0 | 171 | **完全不同** |
| `model.image_encoder` 去 neck vs `model_cp.image_encoder` 去 neck | 0 | 171 | **完全不同** |

因此当前文档中的严谨表述应是:

> **`cellfinder backbone` 用的是 `model` 分支的 backbone 主体（除去 neck）。**  
> 若按我们项目脚本的类外访问写法，也可以记作: **`cellfinder backbone` 对齐 `model.model.image_encoder` 的非-neck部分。**

### 12.6 为什么 `cellfinder` 去掉 neck 后才和 `model` 一致

原因很直接: `cellfinder` 里本来就**没有 neck**。

源码证据:

- `cellSAM_source/cellSAM/AnchorDETR/models/backbone.py:185`
- `cellSAM_source/cellSAM/AnchorDETR/models/backbone.py:190`
- `cellSAM_source/cellSAM/AnchorDETR/models/backbone.py:191`
- `cellSAM_source/cellSAM/AnchorDETR/models/backbone.py:192`
- `cellSAM_source/cellSAM/AnchorDETR/models/backbone.py:219`
- `cellSAM_source/cellSAM/AnchorDETR/models/backbone.py:221`

`ModifiedImageEncoderViT` 只保留:

1. `patch_embed`
2. `blocks`
3. `pos_embed`

没有复制 `image_encoder.neck`。因此:

- 比较 `cellfinder backbone` 时, 必须拿 `model.image_encoder` 去掉 neck 后再比较
- 这也是为什么本地实测结果是:
  - `cellfinder` vs `model(no neck)` 完全一致
  - `cellfinder` vs `model_cp(no neck)` 完全不同

简化理解:

```text
cellfinder backbone
  = model 分支 image_encoder 的 ViT 主体
  = patch_embed + pos_embed + transformer blocks
  ≠ 完整 image_encoder
  ≠ neck
```

### 12.7 官方 checkpoint / 相关权重文件在哪里

当前本地可直接定位到的相关权重路径如下。

#### CellSAM 官方 checkpoint

它们不是 `model / model_cp / cellfinder` 三个独立文件, 而是**一个 CellSAM checkpoint 文件里包含三组 state_dict 前缀**:

- `C:\Users\71069\.deepcell\models\cellsam_v1.2\cellsam_general.pt`
- `C:\Users\71069\.deepcell\models\cellsam_v1.2\cellsam_extra.pt`

本地实测到的前缀为:

1. `model`
2. `model_cp`
3. `cellfinder`

也就是说:

> `model`、`model_cp`、`cellfinder` 是**同一个 CellSAM 官方 checkpoint 内部的三组参数**, 不是三个单独下载的 `.pt` 文件。

#### 其他常用对照权重

- 原始 SAM ViT-B:
  - `d:\AI\paper\CellSam\checkpoints\sam_vit_b_01ec64.pth`
- MedSAM:
  - `d:\AI\paper\CellSam\checkpoints\medsam_vit_b.pth`
  - `d:\AI\paper\CellSam\checkpoints\medsam_vit_b_real.pth`

### 12.8 因此，最严谨的当前理解

把论文描述和发布代码一起看，最稳的结论是:

1. **功能层面**: 论文说检测和分割共享 ViT backbone，这指的是“都基于同一类 ViT 特征管线”
2. **对象层面**: 发布代码并没有把它们做成同一个 Python 实例
3. **权重层面**:
   - `cellfinder backbone` 对齐 `model.image_encoder` 的 ViT 主体
   - `分割推理` 用的是 `model_cp`
4. 因而发布产物更像:
   - `model` 承载检测侧对齐的 ViT 主体
   - `model_cp` 承载官方分割推理分支

### 12.9 CellSAM 分割时实际调用的是谁

官方 `CellSAM` 初始化时:

- `adv_mode = True`
  - `cellSAM_source/cellSAM/sam_inference.py:136`

在 `predict()` 里:

- `mdl = self.model_cp if self.adv_mode else self.model`
  - `cellSAM_source/cellSAM/sam_inference.py:327`

因此, **官方 checkpoint 正常加载后, 分割推理实际走的是 `model_cp`**。

本地实测:

- `get_model()` 加载官方 `cellsam_general.pt` 后:
  - `adv_mode = True`
  - checkpoint 内确实存在 `model_cp.*` 前缀

所以当前应使用的严谨表述是:

> **CellSAM 官方分割推理分支 = `model_cp` (`adv_mode=True`)。**  
> `model` 不是官方默认分割分支。

### 12.10 为什么这会和论文“Stage 2 只训 neck”冲突

如果完全按论文字面理解，你会预期:

- `model` 和 `model_cp`
  应该只在 neck 上不同

但实际对比结果不是这样，而是:

- encoder(no neck): `171/171` 不同
- neck: `6/6` 不同
- decoder: `120/120` 不同
- prompt encoder: `17/17` 不同

所以当前应避免把发布 checkpoint 解释成:

> “`model_cp` = 在 `model` 上只改 neck 得到”

更安全的说法是:

> **论文描述的是两阶段训练策略；但公开发布的 checkpoint 中，`model` 与 `model_cp` 表现为两套全局不同的 SAM 分支，不能再简单等同为“只差 neck 的 stage1/stage2 对”。**

关于“为什么连 prompt encoder / mask decoder 也全变了”，当前只能下到这一步:

1. **事实已证实**: `prompt encoder` 与 `mask decoder` 的 state_dict 项在 `model` 与 `model_cp` 间也全部不同
2. **原因不可证**: 公开仓库没有 Stage 2 训练脚本，因此不能确定
   - `model_cp` 是否由 `model` 继续训练得到
   - 发布 checkpoint 是否保留了训练时的中间态语义
   - 作者是否在发布版里做过额外重打包 / 分支重置

因此文档与论文里都不应写成:

> “Stage 2 只改 neck, 但 prompt/decoder 之所以变化是因为 XXX”

因为 `XXX` 在当前公开证据下不可验证。

### 12.11 结构图: 论文层 vs 发布代码层

#### 论文层 (概念结构)

```text
输入图像
  -> ViT backbone (共享特征)
      -> CellFinder
      -> neck -> prompt encoder -> mask decoder
```

#### 发布代码层 (对象结构)

```text
CellSAM
  ├─ model
  │   └─ image_encoder (ViT + neck)         [与 cellfinder backbone 主体对齐]
  ├─ model_cp
  │   └─ image_encoder (ViT + neck)         [官方分割推理分支]
  └─ cellfinder
      └─ decode_head.backbone.body
          └─ ModifiedImageEncoderViT
              = patch_embed + pos_embed + ViT blocks
              = model.image_encoder 去 neck 的同权重版本
```
