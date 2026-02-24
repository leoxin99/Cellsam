# T16 Baseline Experiment — 方法学审计文档

**日期**: 2026-02-21 14:13  
**提交人**: A2 (Implementation Agent)  
**请 R1 审核**: 实验流程、公平性、数据可靠性

---

## 1. 实验总览

| ID | Method | Eval Type | Input | 权重来源 | 评估脚本 |
|----|--------|-----------|-------|----------|----------|
| E-B1a | Cellpose v4 | E2E (自动) | BF gray | pip (cellpose==4.0.8) | `baseline_eval.py --method cellpose_v4` |
| E-B4 | CellSAM Pretrained | Oracle (GT box) | BF×3 | `get_model()` 原始权重 | `baseline_eval.py --method cellsam_pretrained` |
| E-B5 | SAM ViT-B (vanilla) | Oracle (GT box) | BF×3 | 旧结果 `results_combined.json` | `baseline_eval.py --method medsam`（旧 weights） |
| E-B5b | **MedSAM** | Oracle (GT box) | BF×3 | Zenodo record/10689643 (357MB) | `baseline_eval.py --method medsam` |
| E-B6 | SAMCell (LIVECell) | E2E (自动) | BF gray | GitHub NathanMalta/SAMCell v1 release (348MB) | `samcell_eval.py` |
| E-B7 | Ours Oracle | Oracle (GT box) | BF×3 | `E_phase1_rebalance_l4/best_model.pt` | `comprehensive_eval.py` |
| E-B8 | Ours E2E | E2E (DAPI detect) | BF×3 + DAPI | `E_phase1_rebalance_l4/best_model.pt` + locked_eval | `evaluate_e2e.py` |

**统一评估**: 所有方法的 instance mask 均通过 `compute_all_metrics()` 计算，保证指标口径一致。

---

## 2. 各方法详细设定

### 2.1 Cellpose v4 (E-B1a)

- **版本**: cellpose==4.0.8
- **调用**: `CellposeModel().eval(image, diameter=None, channels=[0,0])`
  - `diameter=None` → 模型自动估计细胞直径
  - `channels=[0,0]` → 灰度模式
- **输入**: BF 通道，归一化 [0,1]→[0,255] uint8
- **预处理**: `cv2.resize` 到 1024×1024

> [!WARNING]
> **发现**: Cellpose v4 检测到 ~255 个小对象 vs GT 10 个。严重过分割——Cellpose 训练集不含大面积 iPSC-CM。
> **公平性**: 使用了 Cellpose 自动直径估计（最佳实践），但也可尝试手动指定 diameter=200。

### 2.2 CellSAM Pretrained (E-B4)

- **调用**: `get_model()` 加载 CellSAM 原始预训练权重（不加载任何 checkpoint）
- **推理**: 使用 `segment_with_boxes()` 统一推理核心 + GT boxes
- **配置**: `mask_threshold=0.5, box_expand=0.1, conflict_policy=argmax_prob`

> [!NOTE]
> 与 Ours Oracle 使用相同推理配置和统一核心，唯一区别是权重。

### 2.3 SAM ViT-B / vanilla (E-B5 旧)

- **权重**: `sam_vit_b_01ec64.pth` (357MB) — Meta 原版 SAM，**非 MedSAM**
- **推理**: 直接调用 `segment_anything` API
- **⚠️ 已被 MedSAM (E-B5b) 替代**

> [!CAUTION]
> `results_combined.json` 中标签为 `"medsam"` 但实际权重是 **vanilla SAM ViT-B**。这是一个标签错误，已被 E-B5b 替代。论文中使用应引用 **SAM ViT-B (vanilla)**。

### 2.4 MedSAM (E-B5b) — ✅ 最新

- **权重**: `medsam_vit_b_real.pth` (357MB)
  - 来源: `https://zenodo.org/record/10689643/files/medsam_vit_b.pth`
  - **已验证**与 SAM ViT-B 权重不同: encoder diff=6295, decoder diff=3275
- **推理**: 使用 `segment_anything` 的 `sam_model_registry["vit_b"]` + GT boxes
- **代码路径**: `baseline_eval.py` → `eval_medsam()`
- **关键实现细节**:
  1. `sam.preprocess(img)` → `sam.image_encoder` → 每图一次编码
  2. 逐 box 调用 `prompt_encoder` + `mask_decoder`
  3. `torch.sigmoid` → `resolve_conflicts` (同我们的 argmax_prob 策略)
  4. 使用 `torch.no_grad()` + `torch.cuda.empty_cache()` 防止 VRAM 泄漏

> [!IMPORTANT]
> **MedSAM 输入**: MedSAM 接收的是 `dataset['image']`，即 BF 通道 replicated 3 次 [BF, BF, BF]。这与 Ours Oracle 的输入完全一致 → 公平比较。

### 2.5 SAMCell (E-B6)

- **模型**: NathanMalta/SAMCell v1 release — `samcell-livecell` (LIVECell 训练)
- **架构**: HuggingFace `SamModel` + fine-tuned decoder → 预测 distance map → watershed
- **输入**: BF 通道灰度 uint8，`SamProcessor.from_pretrained('facebook/sam-vit-base')` 预处理
- **推理**: `crop_size=256` 滑动窗口 → stitching → `watershed` 实例分割
- **代码路径**: `samcell_eval.py` → `SAMCellPipeline.run()`
- **依赖**: `transformers==5.2.0`, `monai`

> [!WARNING]
> **潜在问题**: SAMCell 原始代码 (`pipeline.py`) 使用 `SlidingWindowHelper` 做带 overlap 的 crop stitching。我的实现用了简化版的 averaging stitching。这可能导致 SAMCell 性能略低于其最优。建议如有时间可尝试用原始 `SlidingWindowHelper`。

### 2.6 Ours Oracle (E-B7)

- **权重**: `checkpoints/E_phase1_rebalance_l4/best_model.pt` (Phase1 L4 rebalance)
- **推理**: `comprehensive_eval.py` → `segment_with_boxes()` + GT boxes
- **配置**: `InferenceConfig.default()` = `mask_threshold=0.5, box_expand=0.1, conflict_policy=argmax_prob`
- **数据来源**: `experiments/comprehensive_eval/results.json` (timestamp: 2026-02-12)

### 2.7 Ours E2E (E-B8)

- **权重**: 同上 `E_phase1_rebalance_l4/best_model.pt`
- **检测**: DAPI → `detect_and_create_boxes()` with `locked_eval` profile
  - `min_nucleus_area=1500, max_nucleus_area=20000, merge_coeff=1.4`
- **推理**: `evaluate_e2e.py` → `segment_with_boxes()` + detected boxes
- **数据来源**: `experiments/e2e_evaluation/results.json` (timestamp: 2026-02-21)

> [!NOTE]
> E2E 最初用了错误的 `bf_baseline_full_best.pt` (Oracle PQ=0.058)，导致 PQ=0.062。已修正为 `E_phase1_rebalance_l4/best_model.pt` (Oracle PQ=0.464) → E2E PQ=0.180。

---

## 3. 数据完整性检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| test set 大小 | ✅ | 所有方法均使用 test(73) |
| 指标函数 | ✅ | 统一使用 `compute_all_metrics()` |
| GT mask 来源 | ✅ | 统一从 `AugmentedAllenDataset` 加载 |
| IoU 阈值 | ✅ | PQ@0.5 (TP: IoU≥0.5) |
| 成功样本数 | ✅ | 所有方法 n=73，无失败 |
| VRAM 泄漏 | ✅ | MedSAM 已修复 (`torch.no_grad` + `empty_cache`) |
| 随机种子 | ⚠️ | 确定性推理（无随机操作）— 所有方法 eval 模式 |
| 输入一致性 | ⚠️ | 见下方 §4 |

---

## 4. 公平性分析 ⚠️ 需 R1 审核

### 4.1 Oracle vs E2E 的混合比较

| Method | Eval Type | 备注 |
|--------|-----------|------|
| Cellpose | E2E | 自动检测+分割 |
| CellSAM Pretrained | Oracle | 使用 GT boxes |
| MedSAM | Oracle | 使用 GT boxes |
| SAMCell | E2E | 自动检测+分割(distance map + watershed) |
| Ours Oracle | Oracle | 使用 GT boxes |
| Ours E2E | E2E | DAPI 检测 + 分割 |

> **论文分析建议**: Oracle 和 E2E 结果应分开讨论。Oracle 反映分割质量上限，E2E 反映实际可用性。

### 4.2 输入通道差异

| Method | 输入 | RGB/Gray | 说明 |
|--------|------|----------|------|
| Cellpose | BF→gray | uint8 | BF 单通道转灰度 |
| CellSAM Pretrained | [BF,BF,BF] | float [0,1] | BF replicated 3 通道 |
| MedSAM | [BF,BF,BF] | float [0,1] | 同上 |
| SAMCell | BF→gray | uint8 | BF 单通道转灰度转 BGR |
| Ours | [BF,BF,BF] | float [0,1] | BF replicated 3 通道（数据集已配置） |

> **公平性**: 所有方法均基于 BF（明场）通道。Ours E2E 额外使用 DAPI 做检测但分割仍基于 BF。

### 4.3 MedSAM > Ours Oracle 的解释

MedSAM Oracle PQ=0.576 vs Ours Oracle PQ=0.464 (+24%)。这不是方法论错误，原因分析:

1. **MedSAM 预训练数据**: 100万+ 医学图像上 SAM fine-tuning（Nature Medicine 2024）
2. **我们的训练数据**: 仅 ~200 张 iPSC-CM 图像
3. **两者使用相同 GT boxes 和 conflict resolution** → 比较公平
4. **关键差异**: MedSAM 没有检测能力，不能做 E2E

> **论文写法建议**: "MedSAM demonstrates superior segmentation quality under oracle conditions, benefiting from large-scale medical pretraining. However, it cannot perform end-to-end cell detection, while our pipeline achieves fully automated analysis."

---

## 5. 最终结果表 (论文用)

| Method | Type | PQ | BM-Dice | AJI | Sem.Dice | SQ | RQ |
|--------|------|----|---------|-----|----------|----|----|
| Cellpose v4 | E2E | 0.000 | 0.053 | 0.025 | 0.079 | — | — |
| SAMCell | E2E | 0.000 | 0.008 | 0.004 | 0.014 | — | — |
| CellSAM (pretrained) | Oracle | 0.000 | 0.121 | 0.056 | 0.219 | — | — |
| SAM ViT-B | Oracle | 0.286 | 0.631 | 0.440 | 0.756 | — | — |
| Ours | Oracle | 0.464 | 0.695 | 0.519 | 0.756 | 0.616 | 0.753 |
| MedSAM | Oracle | 0.576 | 0.771 | 0.634 | 0.862 | 0.685 | 0.840 |
| Ours | E2E | 0.180 | 0.567 | 0.338 | 0.642 | 0.544 | 0.305 |

---

## 6. 结果文件索引

| 文件 | 对应实验 |
|------|----------|
| `experiments/baseline_comparison/results_combined.json` | Cellpose, CellSAM pretrained, SAM ViT-B (旧) |
| `experiments/baseline_comparison/results.json` | MedSAM (最新) |
| `experiments/baseline_comparison/per_sample_samcell_livecell.json` | SAMCell |
| `experiments/comprehensive_eval/results.json` | Ours Oracle (Phase1_L4) |
| `experiments/e2e_evaluation/results.json` | Ours E2E (Phase1_L4 + locked_eval) |

---

## 7. 待 R1 决策

1. **SAMCell 简化版 stitching** — 是否需要用原始 SlidingWindowHelper 重跑？（影响可能很小，SAMCell 在大细胞上本质性失败）
2. **SAM ViT-B (vanilla) 是否需要从论文中移除？** — 已有 MedSAM 作为更强的 SAM 系列 baseline
3. **Cellpose 是否尝试手动设 diameter=200？** — 出于公平性考虑
4. **论文中 MedSAM > Ours 如何表述？** — 见 §4.3 建议
5. **StarDist 是否仍需补充？** — 需要独立 TensorFlow 环境
