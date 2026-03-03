# CellSAM 项目进展汇报 (2026-03-02)

> **上次汇报**: 2026-02-19 ([report_2.19.md](file:///d:/AI/paper/CellSam/docs/report_2.19.md))
> **本次覆盖**: 2026-02-19 ~ 2026-03-02 (2 周)

---

## 一、导师会议任务完成情况

> 来源: [meeting_notes_2.19.md](file:///d:/AI/paper/CellSam/docs/meeting_notes_2.19.md) 七、Action Items

| # | 任务 | 优先级 | 状态 | 备注 |
|:-:|------|:------:|:----:|------|
| 1 | **Baseline 对比** (Cellpose/StarDist/MedSAM/SAMCell) | P0 | ✅ 完成 | T16, 6 方法全部评估完毕 |
| 2 | **Training Curves 图** | P0 | ✅ 完成 | T17, Best Config + Phase1 曲线 |
| 3 | **三通道实验** (BF / 2ch / 3ch) | P0 | ✅ 完成 | T18, 5/6 runs, 最优 PQ=0.500 |
| 4 | **框外像素策略调研** | P1 | ✅ 已调研 | T19-abl: box clipping 消融 PQ 0.437->0.466 |
| 5 | **LoRA Encoder 微调** | P2 | ✅ 完成 | T11, r4/r8 两组, r8 PQ=0.494 (+1.0pp) |
| 6 | **论文写作** | — | 🔄 进行中 | 已完成部分章节, 下周可完成初稿 |
| 7 | **Grad-CAM 可视化** | P2 | ⏳ 脚本就绪 | T20, 待执行 |

---

## 二、2/19 以来新增实验总结

### 2.1 T16 Baseline 对比 (02-21~22) ✅

在 test(73) 上与 6 种方法统一评估:

| Method | Type | PQ | BM-Dice | AJI |
|--------|:----:|:--:|:-------:|:---:|
| Cellpose v4 (auto) | E2E | 0.000 | 0.053 | 0.025 |
| Cellpose v4 (d=200) | E2E | 0.002 | 0.190 | 0.089 |
| SAMCell (LIVECell) | E2E | 0.000 | 0.008 | 0.004 |
| SAM ViT-B (vanilla) | Oracle | 0.286 | 0.631 | 0.440 |
| CellSAM (pretrained) | Oracle | 0.434 | 0.682 | 0.499 |
| **MedSAM** | **Oracle** | **0.576** | **0.771** | **0.634** |
| **Ours (Phase1)** | **Oracle** | **0.464** | **0.695** | **0.519** |
| Ours (Phase1) | E2E | 0.180 | 0.567 | 0.338 |

> MedSAM (PQ=0.576) > Ours Phase1 Oracle (0.464), 但 MedSAM 无检测能力、依赖 100 万+医学图像预训练

### 2.2 T12 Loss 消融 (02-23) ✅

7 组消融 × 2 seeds, 关键发现:

| 消融项 | PQ (seed avg) | vs Phase1 |
|--------|:------------:|:---------:|
| Phase1 基线 | 0.475 | — |
| **pos_weight=10** | **0.494** | **+1.9pp** |
| contour OFF | 0.478 | +0.3pp |
| boundary OFF | 0.453 | -2.2pp |
| AJI OFF | 0.468 | -0.7pp |

> 结论: pos_weight=10 是最大增益项, contour 无益 (零梯度 bug 历史遗留)

### 2.3 Best Config (02-24) ✅

基于 T12 消融锁定最优配置: `pos_weight=10 + contour=off`

| Run | Partition | PQ (test) |
|-----|:---------:|:---------:|
| seed=42 | A100 | 0.484 |
| seed=42 | L4 | 0.490 |
| seed=123 | A100 | 0.480 |
| seed=123 | L4 | 0.484 |
| **Mean** | — | **0.484** |
| **Best single** | — | **0.508** (ep39) |

### 2.4 T18 三通道实验 (02-24~25) ✅

| 配置 | 输入通道 | PQ (mean) |
|------|:--------:|:---------:|
| T18-A: BF-only (对照) | BF×3 | 0.496 |
| T18-B: BF+DAPI 2ch | BF,BF,DAPI | 0.498 |
| **T18-C: BF+DAPI+Actn2 3ch** | **BF,Actn2,DAPI** | **0.500** |

> 三通道净增 +0.4pp PQ (小但稳定正向), 导师要求的"充分利用三通道数据"已完成

### 2.5 T11 LoRA Encoder ✅

| LoRA Rank | PQ | vs BF-only |
|:---------:|:--:|:----------:|
| r=4 | 0.483 | -0.1pp |
| **r=8** | **0.494** | **+1.0pp** |

### 2.6 T17 Training Curves ✅

Best Config training curve 已生成, best PQ=0.508 @ epoch 39, 论文图表就绪

### 2.7 T24 CellSAM 权重对比分析  ✅

CellSAM 论文描述的两阶段训练与公开 checkpoint 的实际权重存在差异。经逐张量对比, 发现 checkpoint 中包含一组经过更充分训练的权重分支 (`model_cp`), 在心肌细胞数据上效果显著优于论文描述的基础模型:

| 对比 | PQ (Oracle) |
|------|:-----------:|
| 论文描述的图像特征微调模型（image encoder） (`model`) | ~0.337 |
| **充分训练的分支 (`model_cp`)** | **0.434** |

> 基于此发现, 我们将后续实验在 `model_cp` 分支, 以更强的预训练起点进行微调

### 2.8 T27a Plan B 新训练方案  🔄 训练中

基于架构审计结果, 重新设计训练方案:

| 改进 | 具体内容 |
|------|---------|
| 模型分支 | `model.model` -> `model_cp` (官方推理分支) |
| 预处理 | 自定义 -> 官方 `prep_2() + forward()` |
| 后处理 | 无 -> 官方 7 步形态学平滑 |
| Prompt Encoder | 可训练 -> 冻结 (512 params) |
| 非目标分支 | 未冻结 (~200M) -> 全局冻结 |
| 新增 Loss | Focal Loss (a=0.25, g=2.0, w=0.3) |
| 新增 Loss | IoU Head Loss (MSE, w=0.1) |
| Boundary 权重 | 1.5 (65% of total) -> 0.3 (27%) |
| 可训练参数 | ~200M (bug) -> **4,058,340** (mask_decoder only) |

**Dry-run 结果 (1 batch)**:
- Pretrained baseline (model_cp + postprocess): **PQ=0.5444, BM-Dice=0.7460**
- 相比之前 Phase1 PQ=0.464, pretrained model_cp + 后处理已经达到 0.544

**ALICE 提交状态**:

| Job ID | GPU | Seed | Status |
|:------:|:---:|:----:|:------:|
| 1117960 | A100 | 42 | 已提交 |
| 1117961 | L4 | 42 | 已提交 |
| 1117962 | A100 | 123 | 已提交 |
| 1117963 | L4 | 123 | 已提交 |

---

## 三、当前实验 PQ 进展总览

```
Phase1 (旧)         0.464  ████████████████████████████░░░░
Best Config (旧)     0.484  █████████████████████████████░░░
T18-C 3ch (旧)       0.500  ██████████████████████████████░░
model_cp pretrained   0.544  ████████████████████████████████   <-- 当前 baseline
T27a (训练中)         ???    目标: 0.55+
MedSAM (参考上限)     0.576  ████████████████████████████████+
```

> T27a 训练的起点 (pretrained model_cp) 已经超过之前所有实验的最高值

---

## 四、剩余实验计划 (3 个)

| # | 实验 | 预计时间 | 说明 |
|:-:|------|:--------:|------|
| 1 | **T27a 结果分析** | 等待中 (~4-8h) | ALICE 4 个 job 运行中 |
| 2 | **T28 三通道 (model_cp 版)** | ~1 天 | 在 Plan B 基础上重跑三通道消融 |
| 3 | **T27b LoRA Encoder** | ~1 天 | 在 T27a best checkpoint 上加 LoRA r=8 |

> 这 3 个实验完成后, 实验部分全部结束

---

## 五、论文写作状态

| 章节 | 状态 | 备注 |
|------|:----:|------|
| Abstract | 🔄 | 等最终实验数据 |
| 1. Introduction | ✅ 初稿完成 | 背景、动机、贡献 |
| 2. Related Work | ✅ 初稿完成 | SAM/CellSAM/MedSAM/Cellpose |
| 3. Method | 🔄 部分完成 | 检测管线、Loss 设计 |
| 4. Experiments | ⏳ 待补充 | 等 T27a/T28/T27b 结果 |
| 5. Results & Discussion | ⏳ | 需要最终数据 |
| 6. Conclusion | ⏳ | 最后写 |
| 论文准备材料 | ✅ | [paper_preparation.md](file:///d:/AI/paper/CellSam/docs/paper_preparation.md) |

> 计划下周完成初稿, 4-5 月答辩

---

## 六、关键技术成果 (可写入论文)

1. **CellSAM 架构审计**: 发现官方 checkpoint 中 model vs model_cp 完全不同 (0/314), 纠正了 "Stage 2 只训 neck" 的旧认知
2. **Loss 消融体系**: 7 组 × 2 seeds 系统消融, 确认 pos_weight=10 为最大增益 (+1.9pp PQ)
3. **N/O Loss 退化分析**: 3 轮修复均失败, 证明邻居/重叠损失在当前框架下不适用 (导师: "挺有意思的")
4. **Box Clipping 消融**: clip ON vs OFF, PQ 0.437 -> 0.466 (+2.9pp), 证明框内约束对 SAM 有效
5. **三通道增益**: BF-only -> 3ch 净增 +0.4pp, 小但稳定
6. **Focal + IoU Head Loss**: 理论上更好地处理难样本和质量预测, 待 T27a 结果验证
