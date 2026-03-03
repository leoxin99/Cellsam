# CellSAM 官方推理路径迁移方案 (T25 Plan B)

> **创建日期**: 2026-02-28
> **触发**: T24 审核发现项目使用 `model.model` (PQ≈0) 而非官方 `model_cp` (PQ=0.434)
> **本文档**: 方案 B — 将训练/推理管线从自定义 `segment_with_boxes()` 迁移至官方预处理路径

---

## 1. 问题背景

### 1.1 为什么当初没有采用官方 `predict()`？

根据项目文档 `codex_claude_seg.md` 第八章 (2026-02-09, Codex 审核):

**根因: 多脚本推理不一致**

当时项目存在 **5+ 个不同的推理脚本**，各自独立实现了推理逻辑，互相之间口径不一致:

| 差异项 | 官方路径 (`cellSAM/model.py`) | 自定义路径 (多个脚本) |
|--------|:-:|:-:|
| 冲突裁决 | `np.max` 覆盖 | `first_write` / `last_write` / 混合 |
| 阈值 | 0.4 | 0.5 (硬编码) |
| IoU 过滤 | 有 (`iou < 0.5` 跳过) | 无 |
| 后处理 | 默认关 | 有的开有的关 |
| 指标汇总 | — | 按图 / 按细胞 不统一 |

Codex 的结论是: **"主要问题不是 Oracle vs E2E，而是同一模型在多脚本中推理细节不一致"**

解决方案: 新建 `src/inference/core.py`，实现单一入口 `segment_with_boxes()`，所有脚本统一调用。

**但统一过程中遗漏了两个关键点:**

1. **model.model vs model_cp** — 代码全部写成 `model.model.*`，未察觉 CellSAM 有两套权重分支
2. **预处理管线差异** — 只用了 `sam_preprocess()`，丢弃了官方的 PercentileThreshold / ImageNet normalize / Standardize

### 1.2 `predict()` vs `segment_with_boxes()` 本质区别

```python
# -------- 官方 predict() 的核心调用链 --------
# sam_inference.py L286-395

def predict(self, images, boxes_per_heatmap):
    # Step 1: generate_embeddings() → prep_2() → forward()
    x, paddings = self.generate_embeddings(images)
    
    # prep_2() 内部:
    #   1. Resize(1024)
    #   2. sam_preprocess_pad() — 只做 padding，不做标准化
    #   3. PercentileThreshold() — 去极端亮度 (0-99.5%)
    #   4. self.normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) — ImageNet 标准化
    #   5. Standardize() — kornia min-max 归一化
    
    # forward() 内部:
    #   6. sam_preprocess(div_255=True) — SAM pixel_mean/255, pixel_std/255
    #   7. model_cp.image_encoder() — ★ 使用 Stage 2 权重
    
    # Step 2: 逐框解码 (model_cp)
    mdl = self.model_cp  # ★ adv_mode=True
    for bbox in boxes:
        sparse, dense = mdl.prompt_encoder(boxes=bbox)
        masks, iou = mdl.mask_decoder(image_embeddings=x, ...)
        
        if iou < 0.5: continue  # ★ IoU 过滤
        masks = Sigmoid(masks) > 0.4  # ★ 阈值 0.4
    
    # Step 3: 堆叠合并
    return np.max(thresholded_masks * arange, axis=0)

# -------- 我们的 segment_with_boxes() --------
# inference/core.py L146-272

def segment_with_boxes(model, image, boxes, config):
    # Step 1: 预处理 + 编码 (简化版)
    img_preprocessed = model.sam_preprocess(image)  # 仅 SAM 标准化
    image_embedding = model.model.image_encoder(img_preprocessed)  # ★ Stage 1 权重
    
    # Step 2: 逐框解码 (model.model)
    for box in boxes:
        sparse, dense = model.model.prompt_encoder(boxes=box)
        masks, iou = model.model.mask_decoder(image_embeddings=embedding, ...)
        # ★ 无 IoU 过滤
        pred = Sigmoid(masks)  # ★ 阈值 0.5
    
    # Step 3: argmax_prob 冲突裁决 + box clipping
    return resolve_conflicts(pred_stack, threshold=0.5, policy='argmax_prob')
```

### 1.3 预处理管线 (Preprocessing Pipeline) 详解

"预处理管线"指图像在送入 ViT encoder 之前经历的所有数值变换。不同的预处理会产生完全不同的输入张量，即使使用相同的 encoder 权重也会得到不同的特征。

```
官方路径 — 5 步预处理:

原始图像 [C,H,W] float [0,255]
    │
    ├── ① PercentileThreshold(0, 99.5)
    │     去掉极端亮度值 (用 skimage.exposure.rescale_intensity)
    │     例: [30, 50, 200, 255] → [0.0, 0.12, 1.0, 1.0] (255 被当作异常值截掉)
    │
    ├── ② ImageNet Normalize  
    │     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    │     这是 CellSAM Stage 2 训练时的标准化方式
    │
    ├── ③ Standardize (kornia normalize_min_max)
    │     将每通道线性映射到 [0, 1]
    │
    ├── ④ sam_preprocess(div_255=True)
    │     减去 SAM pixel_mean/255, 除以 SAM pixel_std/255
    │     padding 到 1024×1024
    │
    └── ⑤ model_cp.image_encoder() → 特征

我们的路径 — 1 步预处理:

原始图像 [C,H,W] float [0,1] (dataset 已归一化)
    │
    ├── ① sam_preprocess()
    │     减去 SAM pixel_mean, 除以 SAM pixel_std
    │     padding 到 1024×1024
    │
    └── ② model.model.image_encoder() → 特征
```

---

## 2. 项目代码与 CellSAM 官方的所有差异

### 2.1 推理管线差异

| 差异点 | CellSAM 官方 (`predict()`) | 我们 (`segment_with_boxes()`) | 影响程度 |
|--------|:-:|:-:|:-:|
| **Encoder 权重** | `model_cp` (Stage 2) | `model.model` (Stage 1) | 🔴 致命 |
| **Decoder 权重** | `model_cp.mask_decoder` | `model.model.mask_decoder` | 🔴 致命 |
| **Prompt Encoder 权重** | `model_cp.prompt_encoder` | `model.model.prompt_encoder` | 🔴 致命 |
| PercentileThreshold | ✅ (0-99.5%) | ❌ 无 | 🟡 中 |
| ImageNet Normalize | ✅ `[0.485,0.456,0.406]` | ❌ 无 | 🟡 中 |
| Min-Max Standardize | ✅ kornia | ❌ 无 | 🟡 中 |
| SAM preprocess 参数 | `div_255=True` | `div_255=False` | 🟡 中 |
| IoU 过滤 | ✅ `iou < 0.5 → skip` | ❌ 无 | 🟢 低 |
| Mask 阈值 | 0.4 | 0.5 | 🟢 低 |
| 冲突裁决 | `np.max` 编号覆盖 | `argmax_prob` 概率最大 | 🟢 低 |
| Box clipping | ❌ 无 | ✅ 有 | 🟢 低 (我们的改进) |

### 2.2 训练管线差异

| 差异点 | 官方 (用 predict 路径训练的权重) | 我们的 train.py |
|--------|:-:|:-:|
| 数据预处理 | prep_2 (Percentile + ImageNet + Standardize) | `_normalize_image()` (P1-P99 percentile 截断 → [0,1]) |
| Encoder 调用 | `model_cp.image_encoder` | `model.model.image_encoder` |
| Decoder 调用 | `model_cp.mask_decoder` | `model.model.mask_decoder` |
| 冻结策略 | 未知 (Stage 2 训练细节) | `freeze_encoder=True, freeze_decoder=False` |
| Loss 函数 | 未知 (官方训练细节) | Combined (Dice + BCE + Boundary + AJI) |

### 2.3 数据管线差异

| 差异点 | 官方 CellSAM | 我们 (`augmented_dataset.py`) |
|--------|:-:|:-:|
| 输入像素范围 | [0, 255] float | [0, 1] float (percentile 归一化后) |
| 通道处理 | 直接用原始通道 | BF-only (复制3x) 或语义映射 |
| 数据增强 | 未知 | Albumentations (rotate/flip/elastic/GridDistortion/brightness) |

---

## 3. Pixel Range 0-1 vs 0-255 解释

### 3.1 为什么重要

SAM encoder 的预处理 (`sam_preprocess`) 内部硬编码了 `pixel_mean` 和 `pixel_std`:

```python
# SAM 的预设值 (segment_anything/modeling/sam.py)
pixel_mean = [123.675, 116.28, 103.53]   # 基于 ImageNet [0,255] 范围训练
pixel_std  = [58.395, 57.12, 57.375]
```

如果输入是 `[0,255]` 范围的图像:
```
标准化 = (pixel - 123.675) / 58.395   # 结果约 [-2.1, +2.2]
```

如果输入是 `[0,1]` 范围的图像:
```
标准化 = (pixel - 123.675) / 58.395   # 结果约 [-2.1, -2.1]  ← 几乎恒定!
```

**这就是为什么 `sam_preprocess` 有 `div_255` 参数:**

```python
def sam_preprocess(self, x, div_255=False):
    mean = self.model.pixel_mean         # [123.675, ...]
    std = self.model.pixel_std           # [58.395, ...]
    if div_255:
        mean = mean / 255               # [0.485, ...]
        std = std / 255                  # [0.229, ...]
    x = (x - mean) / std
```

- **官方路径**: 图像先经过 prep_2 处理 (包含 Normalize + Standardize)，最后在 forward() 里调 `sam_preprocess(div_255=True)` — 此时输入已经不再是原始 [0,255]
- **我们的路径**: 数据集输出 [0,1]，直接调 `sam_preprocess(div_255=False)` — 用 [0,255] 范围的 mean/std 去标准化 [0,1] 的输入，数值严重偏移

### 3.2 实际数值对比

假设一个像素值:
- 官方路径 (经过 prep_2 后，值约 0.5): `(0.5 - 0.485) / 0.229 = 0.065`
- 我们的路径 (值约 0.5): `(0.5 - 123.675) / 58.395 = -2.109`

**差距巨大!** 这就是为什么即使复制了 model_cp 权重到 model.model，我们的管线仍然只有 PQ=0.072。

---

## 4. Plan B 方案: 迁移至官方预处理路径

### 4.1 修改清单

#### 4.1.1 `src/inference/core.py` — `segment_with_boxes()`

```diff
 def segment_with_boxes(model, image, boxes, config, device):
+    # 使用官方预处理管线:
+    # 1. PercentileThreshold
+    # 2. ImageNet Normalize  
+    # 3. Standardize (min-max)
+    # 4. sam_preprocess(div_255=True)
+    # 5. model_cp encoder/decoder
     
-    img_preprocessed = model.sam_preprocess(image)
-    image_embedding = model.model.image_encoder(img_preprocessed)
+    from cellSAM.AnchorDETR import transforms as anchorT
+    import torchvision.transforms.v2 as T
+    
+    # Step 1-3: 官方预处理
+    img = anchorT.PercentileThreshold()(image.cpu())
+    img = torch.Tensor(img)
+    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
+    img = normalize(img)
+    img = anchorT.Standardize()(img)
+    
+    # Step 4: SAM 预处理 (div_255=True 因为已标准化)
+    img_preprocessed = model.sam_preprocess(img.unsqueeze(0).to(device), div_255=True)
+    
+    # Step 5: 使用 model_cp (Stage 2 权重)
+    image_embedding = model.model_cp.image_encoder(img_preprocessed)
     
     for box in boxes:
-        sparse, dense = model.model.prompt_encoder(...)
-        masks, iou = model.model.mask_decoder(...)
+        sparse, dense = model.model_cp.prompt_encoder(...)
+        masks, iou = model.model_cp.mask_decoder(...)
```

#### 4.1.2 `src/train.py` — 训练循环

训练循环中有 3 处关键调用需修改:

```diff
 # L269: 图像编码
-img_preprocessed = model.sam_preprocess(images)
-image_embedding = model.model.image_encoder(img_preprocessed)
+# 需要将 dataset 输出从 [0,1] 改回 [0,255] 或在此处加预处理
+img_preprocessed = official_preprocess(images)  # prep_2 等效
+image_embedding = model.model_cp.image_encoder(img_preprocessed)

 # L328-336: 逐框 prompt encoder + mask decoder
-sparse_emb, dense_emb = model.model.prompt_encoder(...)
-low_res_masks, _ = model.model.mask_decoder(...)
+sparse_emb, dense_emb = model.model_cp.prompt_encoder(...)
+low_res_masks, _ = model.model_cp.mask_decoder(...)
```

#### 4.1.3 `src/augmented_dataset.py` — 数据输出格式

当前 `_normalize_image()` 输出 `[0,1]` float。官方路径期望 `[0,255]`。

**两个选项**:
- **A**: 数据集继续输出 [0,1]，在训练循环中乘以 255 再走官方预处理
- **B**: 改 `_normalize_image()` 输出 [0,255] float，让 prep_2 做标准化

推荐选项 A (改动最小):
```python
# train.py 训练循环中
images_255 = images * 255.0  # [0,1] → [0,255]
# 然后走 prep_2 预处理
```

#### 4.1.4 `tools/baseline_eval.py` — CellSAM pretrained 评估

```diff
 def eval_cellsam_pretrained(dataset):
     model = get_model()
-    # 删除旧: model.model.load_state_dict(model.model_cp.state_dict())
-    # 直接使用 model.predict() — 官方路径
+    # 直接用官方 predict() 方法
     result = model.predict(images, boxes_per_heatmap=boxes)
```

### 4.2 风险评估

| 风险 | 等级 | 说明 |
|------|:----:|------|
| **所有现有 checkpoint 作废** | 🔴 高 | 旧 checkpoint 在旧预处理下训练，无法在新预处理下使用 |
| 预处理管线 bug | 🟡 中 | prep_2 涉及多个变换库 (kornia, skimage)，需仔细对齐 |
| 训练时间 | 🟡 中 | T25a + T25b 需在 ALICE 重跑约 10h |
| adapter/LoRA 兼容性 | 🟡 中 | 之前的 adapter 基于 model.model 的特征空间，model_cp 特征空间不同 |
| 行为不可预测 | 🟡 中 | model_cp 的 decoder 已针对 prep_2 预处理训练，但我们的 loss 函数不同 |

### 4.3 与方案 A 对比

| | 方案 A (保持现状) | **方案 B (本方案)** |
|---|:---:|:---:|
| 代码改动 | 0 行 | ~100+ 行 (4 个文件) |
| 已有 checkpoint | ✅ 保留 | ❌ 全部作废 |
| Baseline 数值 | T24 官方路径 PQ=0.434 | 同 |
| Our Best PQ | 0.484 (不变) | 未知 (需重训) |
| 理论上限 | decoder 从 PQ≈0 学起 | decoder 从 PQ=0.434 学起，可能更高 |
| 方法学公平性 | 预处理/权重与 baseline 不同 | 与 baseline 使用完全相同的底座 |

---

## 5. 审核要点

1. **必要性**: 方案 A 已能完成论文 (PQ=0.484 > baseline 0.434)，方案 B 是否值得投入?
2. **风险**: 重跑后若 PQ 反而低于 0.484，如何处理?
3. **时间**: ALICE 排队 + 训练约需 1-2 天，是否来得及?
4. **方法学**: 论文 reviewer 是否会质疑"为什么不用官方预处理"?
