# Fig.1-Fig.3 执行 Brief 与素材清单 2026-03-18

> 作者: Codex (A3)
> 作用: 把 `Fig.1 / Fig.2 / Fig.3` 从“图表规划”推进到可直接执行的 brief
> 配套总表: `docs/paper_writing/communication/figure_plan_2026-03-17.md`

## 0. 通用执行原则

1. 这 3 张图都属于正文主线图，不做装饰性扩写。
2. `Fig.1` 必须使用真实显微数据，不允许 AI 生成“示意细胞”替代证据图。
3. `Fig.2` 和 `Fig.3` 可以用 AutoFigure-Edit 起草，但最终必须人工改成统一风格的可编辑图。
4. 统一交付格式建议：
   - 源文件：`svg`
   - 论文插图：`pdf` 或高分辨率 `png`
5. 字体、配色、箭头样式在三张图之间保持一致。

## 1. Fig.1 执行 Brief

### 1.1 图的唯一任务

让读者一眼明白：这个任务不是普通细胞分割。  
`BF` 边界弱，`DAPI` 只给核位置，`Actn2` 只给心肌细胞结构先验，三者都不能直接替代 whole-cell instance mask。

### 1.2 建议放置位置

- `Chapter 2`
- 最适合放在 `2.2 Imaging Modalities and Structural Cues` 或 `2.3 Why Whole-Cell Instance Segmentation Is Difficult in Cardiomyocytes` 之后

### 1.3 建议版式

采用 **一行四联图 + 右侧或下方三个 callout**：

1. `Brightfield`
2. `Actn2`
3. `DAPI`
4. `GT Instance Mask`

Callout 固定三条：

1. `Weak whole-cell boundaries in brightfield`
2. `Nucleus position does not equal full cell extent`
3. `Actn2 provides structural prior, not direct instance masks`

### 1.4 这张图必须出现的元素

1. 同一视野 crop
2. `BF / Actn2 / DAPI / GT` 四个 panel
3. 至少 1-2 个 elongated / crowded cardiomyocyte 区域
4. 清楚的 panel label
5. 明确指出核与全细胞范围不一致的标注

### 1.5 明确不要出现的内容

1. 不放任何性能数值
2. 不放方法流程箭头
3. 不放过多 sample montage
4. 不用 AI 生成假显微图

### 1.6 素材清单

优先从 A1/A2 已锁定的固定 `test73` 三个样本里挑视觉最清楚的一个：

1. `c395b4de_5500000014_63X_20190816_S1_P17_B2`
2. `385fec2b_5500000013_63X_20190807_S2_P14_C2`
3. `9ce733e7_5500000014_63X_20190816_S2_P14_C3`

需要的底层素材：

1. `BF` raw crop
2. `Actn2` raw crop
3. `DAPI` raw crop
4. 对应 `GT instance mask`
5. 最终用于标注的裁剪坐标

### 1.6a 真实样本候选是什么意思

这里的“真实样本候选”不是示意图素材，而是指：

1. **直接来自当前项目数据集的真实图像样本**
2. 后续可能被选为论文正文 `Fig.1` 的正式 panel
3. 必须满足：
   - 同一视野里能同时看清 `BF / DAPI / Actn2 / GT`
   - 有代表性的 elongated / crowded cardiomyocyte
   - 不要只选“最好看”的孤立细胞，也不要选完全看不清边界的极端失败样本

当前建议把固定 `test73` 三样本作为 **一级候选池**：

1. `c395b4de_5500000014_63X_20190816_S1_P17_B2`
2. `385fec2b_5500000013_63X_20190807_S2_P14_C2`
3. `9ce733e7_5500000014_63X_20190816_S2_P14_C3`

筛选顺序建议：

1. 先看哪个样本里 `Actn2` 纹理最清楚
2. 再看哪个样本里 `DAPI` 与 whole-cell extent 的不一致最容易肉眼说明
3. 最后优先选同时存在 2-3 个相邻 elongated cardiomyocyte 的区域

### 1.6b crop 提取清单是什么意思

这里的“crop 提取清单”是指：  
为了把一张原始大图变成论文 panel，我们需要一次性明确哪些裁剪信息和导出信息要固定下来，避免后面同一张图反复重截、坐标漂移、通道不一致。

建议固定以下字段：

1. `sample_id`
2. `split`
3. `source file path`
4. `crop box`
   - `x_min`
   - `y_min`
   - `width`
   - `height`
5. `channels to export`
   - `BF`
   - `Actn2`
   - `DAPI`
   - `GT mask`
6. `normalization rule`
   - 例如 per-channel min-max 或固定 percentile
7. `output names`
   - `fig1_<sample>_bf.png`
   - `fig1_<sample>_actn2.png`
   - `fig1_<sample>_dapi.png`
   - `fig1_<sample>_gt.png`
8. `annotation note`
   - 这块区域想标什么：弱边界？核与全细胞不一致？Actn2 结构先验？

### 1.6c Fig.1 实际执行表单

建议按下面这张最小表单收集：

| 字段 | 内容 |
|---|---|
| `sample_id` |  |
| `split` | `test73` |
| `crop_box` | `(x_min, y_min, width, height)` |
| `BF export` |  |
| `Actn2 export` |  |
| `DAPI export` |  |
| `GT export` |  |
| `main annotation` |  |
| `backup annotation` |  |

### 1.7 建议制作流程

1. 先从真实数据导出同一区域 4 个 crop
2. 在 `PowerPoint / Figma / Illustrator` 做四联版式
3. 手工加 callout、箭头和 label
4. 导出 `svg + pdf`

### 1.8 图注骨架

`Figure 1. Representative multimodal views and biological priors in the Allen human hiPSC-cardiomyocyte dataset. Brightfield provides the deployment-oriented imaging modality but has weak whole-cell boundaries. DAPI provides a positional prior through nuclear localization, while Actn2 provides cardiomyocyte-specific structural cues. None of these channels alone fully determines whole-cell instance extent, which motivates a biology-prior-guided promptable segmentation pipeline.`

## 2. Fig.2 执行 Brief

### 2.1 图的唯一任务

让读者一眼看懂：本文的方法不是“换一个 loss 训一下”，而是一个  
`domain-audited / prompt-aware / cardiomyocyte-specific` 的完整适配框架。

### 2.2 建议放置位置

- `Chapter 4`
- 最适合放在 `4.1 Released-Checkpoint Audit and Method Starting Point` 前后

### 2.3 建议版式

采用 **左到右流程图**，分成 5 个 stage：

1. `Input handling`
2. `Prompt source`
3. `Released checkpoint audit`
4. `T27a adaptation`
5. `Output and evaluation`

每个 stage 最多 2-3 个节点，不要画成巨型系统图。

### 2.4 这张图必须出现的元素

1. `BF-only replicated to 3 channels`
2. `Oracle GT boxes` 与 `automatic boxes`
3. `official segmentation path = model_cp`
4. `image encoder frozen`
5. `prompt encoder frozen`
6. `mask decoder trainable`
7. `Oracle` 与 `End-to-End` 两种评估出口

### 2.5 可以出现但不应抢主位的元素

1. `DAPI / Actn2` 作为 biology priors
2. `Adaptive Z-line` 作为当前更强自动 prompt
3. `CellFinder` 作为 detector-produced prompt 路线

### 2.6 明确不要出现的内容

1. 不放任何结果数值
2. 不把 `H1bA strict` 画成已验证的主方法块
3. 不把 `T37` 后处理对照塞进主干流程
4. 不把历史 `Best Config` 画成当前方法

### 2.7 素材清单

需要从写作材料里抽出的事实节点：

1. `docs/paper_writing/chapters/ch4_experimental_setup.md`
2. `docs/paper_writing/chapters/ch1_introduction.md`
3. `docs/paper_writing/chapters/ch2_background_related_work.md`
4. `docs/paper_writing/chapters/app_a_checkpoint_audit.md`
5. `docs/paper_writing/communication/method_value_narrative_2026-03-12.md`

建议固定节点文案：

1. `BF-only input handling`
2. `GT / automatic prompts`
3. `released CellSAM audit`
4. `official segmentation path: model_cp`
5. `decoder-only T27a fine-tuning`
6. `Oracle evaluation`
7. `End-to-end evaluation`

### 2.8 建议制作流程

1. 先用 AutoFigure-Edit 起草版式
2. 再手工删减节点，防止过度复杂
3. 最终统一到 thesis 字体和配色
4. 导出 `svg + pdf`

### 2.8a Fig.2 low-fidelity wireframe 文本版

```text
[Input handling]
  BF image
    ↓
  replicate to 3 channels
    ↓
  optional note: DAPI / Actn2 as biological priors

                ┌──────────────────────────────┐
[Prompt source] │  Oracle GT boxes             │
                │  Automatic prompts           │
                │   - Adaptive Z-line          │
                │   - detector-produced boxes  │
                └──────────────────────────────┘
                              ↓

                 [Released CellSAM audit]
                 public checkpoint
                 ├─ model
                 ├─ model_cp  ← official segmentation path
                 └─ cellfinder
                              ↓

                    [T27a adaptation]
                    image encoder frozen
                    prompt encoder frozen
                    mask decoder trainable
                              ↓

                  [Output and evaluation]
                  instance masks
                  ├─ Oracle evaluation
                  └─ End-to-end evaluation
```

这版线框的作用不是给排版，而是先锁节点关系。正式图时可以改成横向 5 stage 布局。

### 2.9 图注骨架

`Figure 2. Overview of the thesis pipeline. Brightfield images are converted into a CellSAM-compatible input representation, while DAPI and Actn2 provide biological priors for prompting and future detector design. Prompts are supplied either from ground-truth boxes under Oracle evaluation or from automatic detectors under end-to-end evaluation. The released CellSAM checkpoint is first audited to identify the official model_cp inference path, after which the T27a mainline adapts the mask decoder while keeping the image encoder and prompt encoder frozen.`

## 3. Fig.3 执行 Brief

### 3.1 图的唯一任务

把 `model / model_cp / cellfinder` 的关系讲清楚。  
这张图的作用不是炫结构，而是让读者停止把 released CellSAM 当成单一路径黑箱。

### 3.2 建议放置位置

- `Chapter 4`
- 或者正文放简版、附录放数字细版

### 3.3 建议版式

采用 **一个 checkpoint 容器 + 三条分支**：

顶层：

- `Released CellSAM checkpoint`

下层三分支：

1. `model`
2. `model_cp`
3. `cellfinder`

再加两类标注：

1. `official segmentation inference -> model_cp`
2. `cellfinder backbone aligns with model.image_encoder (non-neck body)`

底部再放一个小注释框：

- `verifiable implementation facts`
- `hidden training history not fully recoverable`

### 3.4 这张图必须出现的元素

1. `model` 与 `model_cp` 是不同分支
2. `cellfinder` 不对齐 `model_cp.image_encoder`
3. segmentation official path 走 `model_cp`
4. 当前论文只宣称可验证实现事实

### 3.5 可选增强元素

1. 角标写一行：
   - ``model.model` Oracle `test73`: PQ = 0.000`
2. 或正文配一句：
   - Branch A is not segmentation-ready

注意：这条量化结论可以出现，但不要把整张图做成数值表。

### 3.6 明确不要出现的内容

1. 不要把 `171/171`, `177/177`, `120/120` 全塞进主图
2. 不要把“Stage 2 为什么会这样训”画成确定结论
3. 不要画成过深的 module tree

### 3.7 素材清单

需要从这些文档提取事实：

1. `docs/paper_writing/chapters/app_a_checkpoint_audit.md`
2. `docs/paper_writing/paper_preparation.md` 的 `2.1b / 2.1c`
3. `docs/paper_writing/chapters/ch4_experimental_setup.md`
4. A2 2026-03-16 21:00 的 `model.model` baseline 结论

建议固定写入图中的事实标签：

1. `official segmentation path`
2. `single-mask inference`
3. `model vs model_cp globally different`
4. `cellfinder aligns with model non-neck body`
5. `training history not fully recoverable from release`

### 3.8 建议制作流程

1. 先用 AutoFigure-Edit 起一个三分支框图
2. 再人工精简为 audit figure，而不是系统大图
3. 把参数级数字保留给附录表或附录图
4. 导出 `svg + pdf`

### 3.8a Fig.3 low-fidelity wireframe 文本版

```text
                 [Released CellSAM checkpoint]
                   /            |             \
                  /             |              \
             [model]       [model_cp]      [cellfinder]
                |               |               |
                |               |               |
   aligns with non-neck body    |      backbone aligns with
   of cellfinder backbone       |      model.image_encoder (non-neck)
                                |
                official segmentation inference
                                |
                             [single-mask path]
                       multimask_output = False

        [Audit boundary]
        - verifiable implementation facts
        - hidden training history not fully recoverable
```

如果需要更强一点的正文版，可以在 `model` 边上加一个小注：

```text
model.model Oracle test73: PQ = 0.000
```

但这条量化信息更适合做角标，不要变成整张图的主内容。

### 3.9 图注骨架

`Figure 3. Conceptual audit of the released CellSAM checkpoint. The public release contains separate model, model_cp, and cellfinder branches. In the released inference implementation, segmentation is performed through the model_cp branch, while the CellFinder backbone aligns with the non-neck body of model.image_encoder rather than with model_cp. The figure summarizes only the implementation facts that are directly verifiable from the public release; it does not claim full recovery of the hidden training path that produced all branch differences.`

## 4. 现在就能启动的素材收集任务

### 4.1 Fig.1

1. 从固定 `test73` 三样本里挑一个最适合作为 biology-prior panel 的 crop
2. 导出 `BF / Actn2 / DAPI / GT` 四联原始素材
3. 统一 crop 坐标

### 4.2 Fig.2

1. 从 `Ch4` 抽出固定流程节点文案
2. 决定图中是否把 `CellFinder` 放进自动 prompt 子节点
3. 先出一版 low-fidelity wireframe

### 4.3 Fig.3

1. 从附录审计文档抽出 5 条固定事实标签
2. 决定是否在图旁小注 `model.model PQ=0.000`
3. 先出一版“无参数计数”的正文简图

---

## 5. Method Principle Figures (Prism-ready English)

> 说明: 以下 4 张是“原理介绍图”草案，可直接粘贴到 Prism 作为图生成提示或图注草稿。  
> 编号可暂用 `Method-F1` 到 `Method-F4`，后续再映射到正式图号。

### 5.1 Method-F1: CellFinder Design and Detection Principle

**Wireframe (English, Prism-ready)**

```text
[Input image: BF / Actn2 / DAPI]
        |
        v
[CellSAM preprocessing]
        |
        v
[SAMBackbone (ModifiedImageEncoderViT)]
  - patch_embed
  - pos_embed
  - ViT blocks
        |
        v
[Feature map + N learned queries]
        |
        v
[Transformer decoder (L layers)]
        |
        +--> [Class head: cell / non-cell]
        |
        +--> [Box head: (cx, cy, w, h)]

Training path:
[Pred queries] <--> [GT boxes]
        |
        v
[Hungarian matching]
        |
        v
[Loss = Focal CE + L1 + GIoU]

Inference path:
[Top-K predicted boxes]
        |
        v
[Prompt boxes for segmentation]
```

**Caption draft (English, Prism-ready)**

`Figure X. CellFinder detection principle and its role in our pipeline. Cell proposals are produced by a Transformer decoder with learned object queries and optimized with Hungarian matching using Focal classification loss and box regression losses (L1 + GIoU). The detected boxes are then used as prompts for downstream instance segmentation.`

### 5.2 Method-F2: Transformer Components and T27a Freeze Strategy

**Wireframe (English, Prism-ready)**

```text
[Released CellSAM checkpoint]
   |---------------- model (Branch A)
   |---------------- model_cp (official segmentation path)
   |                     |
   |                     +--> [Image encoder (ViT + neck)]   [Frozen]
   |                     +--> [Prompt encoder]               [Frozen]
   |                     +--> [Mask decoder + IoU head]      [Trainable]
   |
   |---------------- cellfinder (detector branch)

Segmentation forward:
[BF -> replicate to 3 channels]
        + [Box prompts]
        |
        v
[model_cp embeddings]
        |
        v
[Mask decoder]
        |
        v
[Single-mask output (multimask_output=False)]
```

**Caption draft (English, Prism-ready)**

`Figure X. Transformer-based component roles in released CellSAM and parameter update scope in T27a. Segmentation follows the audited model_cp path. During T27a fine-tuning, the image encoder and prompt encoder are frozen, while the mask decoder (with IoU head supervision) is updated under single-mask inference.`

### 5.3 Method-F3: Domain-Audited and Prompt-Aware System Design

**Wireframe (English, Prism-ready)**

```text
[Allen image: BF, Actn2, DAPI]
        |
        +--> [Segmentation input handling]
        |      BF -> [BF, BF, BF]
        |
        +--> [Biology-prior prompt cues]
               DAPI (localization prior)
               Actn2 (cardiomyocyte structure prior)

[Prompt source]
   +--> Oracle: GT boxes
   +--> Automatic: Adaptive / CellFinder / Hybrid(H1b)
        |
        v
[T27a segmentation on audited model_cp path]
        |
        v
[Instance masks + postprocess]
        |
        +--> [Oracle metrics: PQ, BM-Dice, AJI, SQ/RQ]
        +--> [E2E metrics: PQ, F1/RQ, BM-Dice]

[Oracle-E2E gap analysis -> bottleneck diagnosis]
```

**Caption draft (English, Prism-ready)**

`Figure X. Domain-audited and prompt-aware cardiomyocyte segmentation framework. The method couples an audited model_cp segmentation path with biology-prior-informed prompting (DAPI/Actn2) and explicitly separates Oracle and end-to-end evaluation routes to diagnose prompt-quality bottlenecks.`

### 5.4 Method-F4: T27a Training Method and Optimization Flow

**Wireframe (English, Prism-ready)**

```text
[Initialize from released model_cp checkpoint]
        |
        v
[Freeze image encoder + prompt encoder]
[Train mask decoder (+ IoU head)]
        |
        v
For each epoch:
  For each mini-batch (I, M_gt, B_gt):
    X = replicate(BF) -> 3-channel
    prompts = GT boxes
    M_pred = model_cp(X, prompts, single-mask)
    L_main = Dice+BCE + Boundary + AJI + Focal
    L_iou  = MSE(iou_pred, iou_target)
    L_total = L_main + λ_iou * L_iou
    Update decoder parameters (AdamW)

  Validate on Oracle split:
    record PQ / BM-Dice / AJI / RQ(F1)
  Early stop by validation PQ
  Save best checkpoint W*
        |
        v
[Locked test73 evaluation + E2E evaluation with automatic prompts]
```

**Caption draft (English, Prism-ready)**

`Figure X. T27a decoder-only training and evaluation workflow. Starting from the released model_cp checkpoint, training uses BF-only replicated input and GT-box prompts, with a combined segmentation loss plus IoU-head regression. Model selection is performed by validation PQ, followed by locked test-set and end-to-end evaluations.`
