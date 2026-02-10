# CellSAM 心肌细胞分割优化方案 - Claude & Codex 联合版

> **创建日期**: 2026-02-09
> **作者**: Claude (Antigravity) + Codex 5.3 协作
> **目的**: 供 Codex 审核评估

---

## 一、背景与问题

### 1.1 核心挑战

心肌细胞 (hiPSC-CM) 具有独特的分割挑战：
- **细长形状**: 增加粘连面积
- **Z线干扰**: 边界模糊
- **相互粘连**: 细胞间无明显间隙

### 1.2 当前训练问题

| # | 问题 | 现状 | 影响 |
|---|------|------|------|
| 1 | 训练-推理 Dice 差距大 | Direct-Match=0.71, Best-Match=0.59 | 训练无效提升推理 |
| 2 | 邻居细胞侵占 | 无惩罚，每个 cell 独立训练 | 粘连处边界不准 |
| 3 | 重叠像素归属 | "先到先得"，顺序依赖 | 不公平分配 |
| 4 | Loss 权重不合理 | pos_weight=10，前景占60% | 过度预测前景 |
| 5 | 早停只看 Dice | 不看 PQ 或侵占率 | 掩盖分割问题 |
| 6 | **PQ 显示 0.0** | 可能有 bug | 需验证修复 |

### 1.3 Direct-Match vs Best-Match

```
Direct-Match (训练时):
  - 每个 box 对应固定 cell_id
  - Dice = compare(Pred_box_i, GT_cell_i)
  - 不惩罚侵占邻居

Best-Match (推理时):
  - 每个 GT 找全局最佳匹配
  - 如果 Pred 侵占邻居，IoU 下降
  - 更真实反映分割质量
```

---

## 二、解决方案

### 2.1 邻居侵占损失 L_neighbor (来自 Codex)

**目标**: 惩罚预测侵占相邻细胞区域

```python
# 公式
p_k = sigmoid(logit_k)           # 当前细胞预测 (0-1)
y_k = (mask == k)                # 当前细胞 GT
n_k = (mask > 0) & (mask != k)   # 邻居细胞区域

L_neighbor(k) = mean(n_k * p_k^gamma)
```

**参数调整** (Claude 建议更保守):

| 参数 | Codex 建议 | Claude 调整 | 原因 |
|------|-----------|------------|------|
| gamma | 2.0 | **1.5** | 避免训练不稳定 |
| lambda_neighbor | 0.4 | **0.3** | 保守起步 |

### 2.2 跨实例互斥损失 L_overlap (来自 Codex)

**目标**: 防止同一像素被多个实例高置信占据

```python
# 公式
S(x) = sum_k p_k(x)  # 像素 x 上所有预测的置信度之和
L_overlap = mean(ReLU(S - 1 - margin)^2)
```

**参数**:
- margin = 0.05 (允许轻微重叠)
- lambda_overlap = 0.1 (Claude 从 0.15 降低)

### 2.3 总损失公式

```
L_total = L_base + λ_neighbor * L_neighbor + λ_overlap * L_overlap

L_base = Dice + BCE*pos_weight + Boundary*0.5 + AJI*0.2
```

### 2.4 训练稳定性 (来自 Codex)

- 前 5 epoch 对 neighbor_weight/overlap_weight 线性 warmup
- 保持 box_expand=0.1 先不动

### 2.5 验证指标 (来自 Codex)

| 指标 | 计算公式 | 用途 |
|------|---------|------|
| **IntrusionRate** | \|pred∩other_cells\| / \|pred_fg\| | 监控侵占 |
| **ConflictRate** | \|Σpred_bin≥2\| / \|Σpred_bin≥1\| | 监控重叠 |

---

## 三、关键设计决策

### 3.1 训练验证的作用

```
训练验证 (Validation) 的目的:
  1. 监控模型在未见数据上的表现
  2. 作为早停 (Early Stopping) 的判断依据
  3. 选择最佳模型 checkpoint

当前: 早停看 Dice, 不看 PQ
建议: 切换到 PQ 或 Best-Match Dice
```

### 3.2 Box Clipping 策略 ⚠️ [已更正]

> **实验验证**: Claude 执行了 box clipping 实验 (`tools/experiment_box_clipping.py`)
> **结论**: **必须保留 clipping**，SAM 确实在 box 外有预测

| 阶段 | Box 来源 | Clipping 策略 |
|------|---------|--------------|
| **训练** | GT 框 | ✅ **保留** - 实验证明 SAM 在 box 外预测值达 0.52 |
| **推理** | DAPI/GT 框 | ✅ **保留** - 防止框外错误预测 |

**实验数据 (2026-02-09)**:
```
总 box 数量: 60
Box 外最大预测值: 0.5230 (超过 0.5 阈值!)
平均 Dice (有 clipping): 0.1655
平均 Dice (无 clipping): 0.0613
Dice 差异: -10.4%
```

**结论**: Codex 关于"不应移除 clipping"的判断正确。

### 3.3 配置文件

```yaml
# 推荐配置
loss:
  pos_weight: 2.0             # 原 10.0
  boundary_weight: 1.5        # 原 0.5
  use_neighbor_exclusion: true
  neighbor_weight: 0.3
  neighbor_gamma: 1.5
  use_overlap_exclusion: true
  overlap_weight: 0.1
  overlap_margin: 0.05
  neighbor_dilation: 3        # 邻居搜索膨胀半径 (Claude 补充)

training:
  use_pq_early_stop: true     # 切换到 PQ 早停
```

---

## 四、待执行任务

### 4.1 ~~最高优先~~ → Phase 2

- [ ] **实现 L_neighbor** (邻居侵占损失)
- [ ] **实现 L_overlap** (重叠互斥损失)
- [ ] **添加 IntrusionRate/ConflictRate 日志**
- [ ] **前5 epoch warmup 逻辑**

### 4.2 待验证/修复

- [x] **PQ 为什么显示 0.0** - Phase 0 已修复 (compute_pq_unified)
- [x] **验证时 Best-Match Dice 低 (0.13)** - 已确认是 box clipping 行为

### 4.3 参数调整 → Phase 1 ⭐ (详见第十二章)

- [ ] pos_weight: 10 → 2 (注: 动态限制使其实际影响有限)
- [ ] boundary_weight: 0.5 → 1.5
- [ ] contour_weight: 0.1 → 0.3
- [ ] 早停切换到 PQ
- [ ] E2E Smoke Test 基线数据收集

### 4.4 训练测试

- [ ] 本地 10% 数据验证
- [ ] ALICE 全量训练
- [ ] 对比 Best-Match Dice 提升

---

## 五、验收标准

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| Best-Match Dice | 0.59 | **≥0.65** |
| PQ@0.5 | 0.34 (或0?) | **≥0.42** |
| IntrusionRate | 未测 | **<10%** |
| ConflictRate | 未测 | **<5%** |
| Train vs Inference Gap | 17% | **<5%** |

---

## 六、代码修改位置

| 文件 | 修改内容 |
|------|---------|
| `src/train.py:train_one_epoch` | 添加 L_neighbor, L_overlap 计算 |
| `src/train.py:validate` | 添加 IntrusionRate, ConflictRate 日志 |
| `src/config/*.yaml` | 添加 neighbor/overlap 配置项 |

---

## 七、请 Codex 审核

1. **L_neighbor 公式和参数是否合理?**
2. **L_overlap 的 margin=0.05 是否合适?**
3. **warmup 5 epoch 是否足够?**
4. **还有其他遗漏的问题吗?**

---

## 八、[Codex新增 | 2026-02-09] 训练/推理冲突专项审核与全套整改方案

> **标注说明**: 本章节由 Codex 独立撰写，和上文 Claude 内容分离，不覆盖原文。  
> **建议执行角色**: `Segmentation Infra + Eval Owner`（训练-推理一致性负责人）。

### 8.1 审核结论

#### 8.1.1 设计上合理的部分（确认）

你当前的核心设计是合理的：
- **训练/验证（Oracle）使用 GT 框**，用于衡量模型在理想 prompt 下的分割上限。
- **外部端到端评估（E2E）使用 DAPI 框**，用于衡量真实可部署效果。

这两条路线应该并存，且必须长期保留。

#### 8.1.2 需要修正的关键点（审核结论）

1. 上文“训练阶段移除 box clipping”的建议不成立。  
当前训练逻辑已围绕 clipping 设计，直接移除会破坏既有损失假设。

2. 上文“推理用 DAPI 框后再用 GT 框 clipping”不适合作为标准流程。  
GT 在真实推理不可用，这会导致评估信息泄露；只能在专门诊断实验中单独使用。

3. 目前主要问题不是“GT vs DAPI 双路线”，而是**同一模型在多脚本中推理细节不一致**，导致指标不可比。

### 8.2 训练/推理冲突问题清单（证据化）

| 冲突项 | 现状证据 | 风险 |
|---|---|---|
| 冲突像素归属规则不一致 | `src/train.py:551`（先写保留）；`src/inference/pipeline.py:132`（先写保留）；`cellSAM_source/cellSAM/sam_inference.py:391`（`np.max` 后编号覆盖）；`tools/comprehensive_eval.py:157`/`tools/evaluate_e2e.py:228`（后写覆盖） | 同一 checkpoint 在不同脚本结果不一致，且对 box 顺序敏感 |
| 阈值与 SAM 过滤不一致 | 训练验证常用 `>0.5`（`src/train.py:540`）；CellSAM 路径使用 `mask_threshold=0.4` 与 `iou_threshold=0.5`（`cellSAM_source/cellSAM/sam_inference.py:128`, `cellSAM_source/cellSAM/sam_inference.py:129`, `cellSAM_source/cellSAM/sam_inference.py:350`） | 指标差异混入流程差异，无法定位模型真实增益 |
| 后处理口径不一致 | `src/inference/pipeline.py` 默认开平滑/尺寸过滤（`src/inference/pipeline.py:58`, `src/inference/pipeline.py:59`, `src/inference/pipeline.py:120`, `src/inference/pipeline.py:126`）；训练内验证无同级后处理 | 训练验证与外部评估不对齐 |
| 指标汇总口径不一致 | 训练验证 Best-Match 以图像均值汇总（`src/train.py:582`）；标准评估以细胞均值汇总（`tools/standardized_inference.py:179`, `tools/standardized_inference.py:195`） | 数值不可直接对比 |
| checkpoint/adapter 加载不一致 | 训练保存 `model_state_dict` 与 `adapter_state_dict`（`src/train.py:697`, `src/train.py:699`）；部分推理路径未统一处理 adapter 或字典格式 | E30/E32 等 adapter 实验可能被低估或加载异常 |
| 配置与实现漂移 | P1 配置含 `train_subset`（如 `src/config/bf_instance_p1_20260205.yaml:19`），但训练主流程未消费该字段（`src/train.py:150`） | 配置与实际运行不一致，影响可复现性 |
| `box_expand` 双重来源 | 外层训练/验证来自 config（`src/train.py:669`）；loss 内部仍硬编码 expand=0.1（`src/losses/combined.py:428`） | 参数控制混乱，调参失真 |

### 8.3 分析过程（可复查）

1. 追踪训练入口：`scripts/train_instance_20260205.sh -> src/train.py`。  
2. 追踪训练内验证实现：`src/train.py:validate`。  
3. 追踪标准评估入口：`tools/standardized_inference.py -> segment_cellular_image`。  
4. 追踪统一推理入口：`tools/run_inference.py -> src/inference/pipeline.py`。  
5. 追踪其它评估脚本：`tools/comprehensive_eval.py`、`tools/evaluate_e2e.py` 的手写推理逻辑。  
6. 对比四类核心差异：冲突裁决、阈值过滤、后处理、指标汇总。  
7. 对比模型加载路径：checkpoint 结构与 adapter 加载一致性。

### 8.4 全套解决方案（保留双路线，统一推理内核）

#### 8.4.1 目标

- 保留 `Oracle(GT boxes)` 与 `E2E(DAPI boxes)` 双路线。
- 除 box 来源外，**统一推理内核与指标口径**。
- 让“分割能力差距”和“检测框质量差距”可分解、可解释。

#### 8.4.2 统一架构

1. 新建统一推理核心（建议）：`src/inference/core.py`  
实现单一入口 `segment_with_boxes(...)`，包含：
- 逐框解码
- 阈值与可选 SAM iou 过滤
- 可选 box clipping
- 冲突像素裁决（统一策略）
- 可选后处理
- 返回 instance mask + 中间统计

2. 统一冲突裁决策略（建议默认）
- 使用“**概率最大归属**”策略：每像素取置信度最高实例。  
- 避免先写/后写/编号覆盖导致的顺序依赖。

3. 统一参数来源
- 在 config 增加 `inference:` 小节，集中管理：  
`mask_threshold`、`use_sam_iou_filter`、`sam_iou_threshold`、`apply_postprocess`、`validate_size`、`box_expand`、`conflict_policy`。

4. 统一 checkpoint/adapter loader
- 抽取 `src/utils/checkpoint.py`：统一处理  
`raw state_dict`、`{'model_state_dict':...}`、`adapter_state_dict`。

5. 统一指标实现
- 抽取 `src/metrics/instance_metrics.py`：  
`best_match_dice_per_cell`、`best_match_dice_per_image`、`pq`、`aji`。  
- 训练验证与外部评估共同调用，杜绝重复实现漂移。

#### 8.4.3 文件级实施清单

| 文件 | 动作 |
|---|---|
| `src/inference/core.py` | 新增统一推理内核 `segment_with_boxes` |
| `src/train.py` | `validate` 改为调用 `segment_with_boxes`；增加统一指标调用 |
| `src/inference/pipeline.py` | 改为薄封装，内部调用 `segment_with_boxes` |
| `tools/standardized_inference.py` | 替换直接 `segment_cellular_image` 调用，改用统一内核 |
| `tools/comprehensive_eval.py` | 删除手写逐框推理，改用统一内核 |
| `tools/evaluate_e2e.py` | 删除手写逐框推理，改用统一内核（仅保留 DAPI 框生成） |
| `src/losses/combined.py` | 移除硬编码 expand，统一读配置传参 |
| `src/config/*.yaml` | 补全统一 `inference` 配置块 |

### 8.5 分阶段落地计划

#### Phase 0（半天）
- 固定规范文档：定义 Oracle/E2E 双看板与统一口径。

#### Phase 1（1天）
- 实现 `src/inference/core.py`，接入 `src/inference/pipeline.py`。

#### Phase 2（1天）
- 接入 `src/train.py validate` 与 `tools/standardized_inference.py`。

#### Phase 3（1天）
- 接入 `tools/comprehensive_eval.py`、`tools/evaluate_e2e.py`。
- 合并指标实现与 checkpoint/adapter loader。

#### Phase 4（半天）
- 回归测试、指标对齐验收、更新文档。

### 8.6 验收标准（必须满足）

1. 同一模型 + 同一 boxes 输入，在不同脚本得到同一 instance mask（允许浮点微小差异）。  
2. 调换 box 顺序后，结果变化可控（或严格不变，取决于最终 conflict_policy 设计）。  
3. 训练内 Oracle 验证与外部 Oracle 标准评估，在同配置下指标差异 < 0.5%。  
4. E30/E32 adapter 实验加载流程一致，有明确日志确认 adapter 权重已加载。  
5. 报告固定输出三项：`Oracle(GT)`、`E2E(DAPI)`、`Gap(Oracle-E2E)`。

### 8.7 风险与回退

| 风险 | 说明 | 回退策略 |
|---|---|---|
| 统一内核改动面大 | 影响训练验证与多评估脚本 | 分阶段切换，保留旧入口一周并做 A/B 对照 |
| 指标短期波动 | 口径统一后数值会与历史不完全一致 | 记录旧口径与新口径双报告，过渡期并行 |
| adapter 历史 checkpoint 兼容性 | 旧 checkpoint 字段不全 | loader 增加容错并打印告警 |

### 8.8 与上文方案关系（对齐说明）

1. 上文 `L_neighbor` / `L_overlap` 仍可继续推进。  
2. 但必须先完成本节“一致性治理”，否则新增损失的收益会被流程噪声掩盖。  
3. 推荐执行顺序：  
`先统一推理口径 -> 再上新损失 -> 再做消融`。

---

## 八-A、[Claude审核 | 2026-02-09] Codex 第八章审核意见

### 完全同意的部分 ✅

1. **8.1.1 Oracle/E2E 双路线**: 正确，这是标准做法
2. **8.2 冲突问题清单**: 发现了多处一致性问题
3. **8.4.2 概率最大归属**: 比"先到先得"更公平
4. **8.8 执行顺序**: "先统一口径 → 再上新损失" 完全正确

### Codex 对 Claude 的修正 (已接受) ✅

1. **Box Clipping**: Codex 正确，已通过实验验证
2. **GT 框 clipping 在推理中**: Codex 正确，会导致信息泄露

### 时间估算评估

| 阶段 | Codex 估计 | Claude 评估 |
|------|-----------|------------|
| Phase 0-4 总计 | 4天 | 约5-6天 |

---

## 九、[Codex新增 | 2026-02-09] CellSAM 多细胞/粘连分割机制源码证明

> **标注说明**: 本章节由 Codex 独立补充，和上文 Claude 内容分离。  
> **证据优先级**: `源码实现 > README/论文概述`。

### 9.1 结论（先给出）

1. CellSAM 在一张图上是“**逐框分割 + 后合并**”机制，不是一次性全局联合分配实例。  
2. 在 `segment_cellular_image` 这条路径中，多实例冲突像素通过 `np.max` 合并裁决。  
3. 因此粘连区域的最终归属，既受单框解码质量影响，也受后处理合并规则影响。

### 9.2 源码证据链（可逐行复查）

1. 标准评估脚本调用 `segment_cellular_image`  
`tools/standardized_inference.py:23`  
`tools/standardized_inference.py:170`

2. `segment_cellular_image` 把框传给 `model.predict`  
`cellSAM_source/cellSAM/model.py:115`  
`cellSAM_source/cellSAM/model.py:166`  
`cellSAM_source/cellSAM/model.py:171`

3. `predict` 的内部流程是按框循环  
`cellSAM_source/cellSAM/sam_inference.py:286`  
`cellSAM_source/cellSAM/sam_inference.py:320`  
`cellSAM_source/cellSAM/sam_inference.py:329`

4. 每个框调用 SAM 的 prompt 编码与 mask 解码  
`cellSAM_source/cellSAM/sam_inference.py:333`  
`cellSAM_source/cellSAM/sam_inference.py:339`

5. 每个框得到二值 mask 后堆叠，再合并  
`cellSAM_source/cellSAM/sam_inference.py:359`  
`cellSAM_source/cellSAM/sam_inference.py:384`  
`cellSAM_source/cellSAM/sam_inference.py:388`  
`cellSAM_source/cellSAM/sam_inference.py:391`

### 9.3 与论文/README的关系

1. README 给出的是 API 级说明（如何调用 `segment_cellular_image`）。  
`cellSAM_source/README.md:20`

2. README/论文没有细化“粘连冲突像素如何裁决”的每步实现。  
该细节来自源码，以上证据链可直接复核。

### 9.4 复核方法（防止主观解读）

1. 用 `rg -n "segment_cellular_image|predict\\(|prompt_encoder|mask_decoder|thresholded_masks_summed|np.max"` 在 `cellSAM_source` 检索。  
2. 对照本节列出的行号，逐个确认调用顺序。  
3. 若后续升级 CellSAM 版本，需重新跑第 1 步并更新本节证据行号。

---

## 十、[Codex新增 | 2026-02-09] 原生 CellSAM 与自定义 Pipeline 差异证明（并排）

> **标注说明**: 本章节由 Codex 独立补充，和上文 Claude 内容分离。  
> **用途**: 解释为什么“同一 checkpoint 不同脚本分数可能不一致”。

### 10.1 两条推理路径是什么

1. 原生路径（CellSAM 官方实现）
- 入口常见于标准评估：`tools/standardized_inference.py:23`, `tools/standardized_inference.py:170`
- 核心 API：`cellSAM_source/cellSAM/model.py:115` (`segment_cellular_image`)

2. 自定义路径（项目统一推理模块）
- 入口常见于项目推理脚本：`tools/run_inference.py:22`, `tools/run_inference.py:124`
- 核心 API：`src/inference/pipeline.py:52` (`run_sam_inference`)

### 10.2 源码级并排对照

| 对照项 | 原生 CellSAM 路径 | 自定义 Pipeline 路径 | 影响 |
|---|---|---|---|
| 核心入口 | `segment_cellular_image` (`cellSAM_source/cellSAM/model.py:115`) | `run_sam_inference` (`src/inference/pipeline.py:52`) | 两套逻辑并存 |
| 二值阈值 | `mask_threshold=0.4` (`cellSAM_source/cellSAM/sam_inference.py:128`, `cellSAM_source/cellSAM/sam_inference.py:359`) | 固定 `>0.5` (`src/inference/pipeline.py:117`) | 前景面积、粘连概率不同 |
| SAM iou 过滤 | 有，`iou_threshold=0.5` 且低于阈值直接跳过 (`cellSAM_source/cellSAM/sam_inference.py:129`, `cellSAM_source/cellSAM/sam_inference.py:350`) | 无同级 iou 过滤（代码未见） | 实例保留数量不同 |
| 后处理默认 | `segment_cellular_image` 默认 `postprocess=False` (`cellSAM_source/cellSAM/model.py:119`)；仍做 `fill_holes_and_remove_small_masks` (`cellSAM_source/cellSAM/model.py:176`) | 默认 `apply_postprocess=True` + `smooth_boundary` + `keep_largest_component` + 尺寸过滤 (`src/inference/pipeline.py:58`, `src/inference/pipeline.py:120`, `src/inference/pipeline.py:123`, `src/inference/pipeline.py:126`) | 边界形态与实例数量差异大 |
| 多实例冲突裁决 | 堆叠后 `np.max` (`cellSAM_source/cellSAM/sam_inference.py:388`, `cellSAM_source/cellSAM/sam_inference.py:391`) | 先写入像素保留 `instance_mask==0` (`src/inference/pipeline.py:132`) | 对重叠区域归属规则不同 |
| checkpoint 加载 | 常见脚本自行解析 `model_state_dict`（例如 `tools/standardized_inference.py`） | `load_model` 直接 `model.load_state_dict(state_dict)` (`src/inference/pipeline.py:44`, `src/inference/pipeline.py:45`) | 对字典型 checkpoint 兼容性风险 |
| 框来源（常见用法） | 标准评估常用 GT 框输入 `segment_cellular_image` | 项目脚本常用 DAPI 检测框：`detect_and_create_boxes` (`tools/run_inference.py:120`) | Oracle 与 E2E 是不同任务，不能混比 |

### 10.3 结论（针对“是否冲突”）

1. 这两条路径不是简单“同函数不同入口”，而是确实存在阈值、过滤、后处理、合并规则差异。  
2. 因此同一模型在两条路径上出现分数不一致是预期现象，不足以直接判定训练是否有效。  
3. 要比较模型本身优劣，必须先固定同一路径、同一配置口径。

### 10.4 最小化误差的执行建议

1. 保留双任务评估（Oracle=GT框，E2E=DAPI框），但每个任务内部只能用一条固定推理内核。  
2. 在报告中强制注明：推理入口、阈值、是否启用 iou 过滤、后处理开关、冲突裁决策略。  
3. 不再混用“原生路径分数”和“自定义路径分数”做横向结论。

---

## 十一、[Codex新增 | 2026-02-09] Oracle + E2E 统一评估报告模板

> **标注说明**: 本章节由 Codex 独立补充，和上文 Claude 内容分离。  
> **用途**: 统一实验记录格式，确保不同实验可横向对比与复现。

```md
## [Experiment ID] 统一评估报告（Oracle + E2E）

### 1. 基本信息
- 日期: YYYY-MM-DD
- 模型/Checkpoint: `...`
- 配置文件: `src/config/...yaml`
- 数据划分: train/val/test = ...（固定 split 文件名）
- 代码版本: commit `...`

### 2. 评估任务定义
- Oracle 任务:
  - Box 来源: GT boxes
  - 目的: 评估分割上限（解耦检测误差）
- E2E 任务:
  - Box 来源: DAPI boxes
  - 目的: 评估真实部署效果（检测+分割联合）

### 3. 推理口径（必须完整记录）
- 推理入口: `...`（例如 `src/inference/core.py::segment_with_boxes`）
- mask_threshold: ...
- use_sam_iou_filter: true/false
- sam_iou_threshold: ...
- apply_postprocess: true/false
- validate_size: true/false
- box_expand: ...
- conflict_policy: `...`（如 `argmax_prob` / `first_write` / `npmax_label`）
- checkpoint加载方式: `model_state_dict`/raw + adapter 是否加载

### 4. 指标口径（必须完整记录）
- Best-Match Dice:
  - per-cell mean: ...
  - per-image mean: ...
- PQ@0.5: ...
- AJI: ...
- IntrusionRate: ...
- ConflictRate: ...
- 汇总方式: mean ± std（注明按图像还是按细胞）

### 5. 结果总表
| Task | Dice(cell) | Dice(image) | PQ@0.5 | AJI | Intrusion | Conflict |
|---|---:|---:|---:|---:|---:|---:|
| Oracle (GT) | ... | ... | ... | ... | ... | ... |
| E2E (DAPI) | ... | ... | ... | ... | ... | ... |
| Gap (Oracle-E2E) | ... | ... | ... | ... | ... | ... |

### 6. 诊断结论
- 模型分割能力（Oracle）是否提升: ...
- 端到端能力（E2E）是否提升: ...
- 主要瓶颈在检测还是分割: ...
- 是否存在口径变化风险: ...

### 7. 可复现实验命令
- Oracle:
  - `python ...`
- E2E:
  - `python ...`
- 评估:
  - `python ...`
```

---

## 十二、[Claude新增 | 2026-02-10] Phase 1 实施方案：Loss 权重调整 + PQ 早停

> **标注说明**: 本章节由 Claude (Antigravity) 撰写，基于代码审查结论。
> **前置**: Phase 0（统一推理口径）已完成，文件整理（A0→A→B）已落地。
> **目标**: 调整 Loss 权重参数 + 启用 PQ 早停。

### 12.1 代码审查关键发现

#### 12.1.1 `pos_weight` 配置值实际无效 ⚠️

**现状**: `base.yaml` / `bf_instance_p2_20260205.yaml` 均设 `pos_weight=10.0`

**发现**: `CombinedLoss.forward()` (`src/losses/combined.py:442-443`) 已有动态限制：

```python
dyn_pos_weight = min(n_neg / n_pos, self.pos_weight)
```

心肌细胞数据前景占 ~60%（细胞尺寸大），因此：
- `n_neg / n_pos ≈ 0.4 / 0.6 ≈ 0.67`
- `min(0.67, 10.0) = 0.67` → 配置值 10.0 根本用不到

**结论**: 把 `pos_weight` 从 10→2 对大多数样本**无实际效果**。仅在极少数小细胞（前景 < 33%）时才有区别。改动主要是文档清晰度价值。

#### 12.1.2 `base_weight` 触底效应

**现状**: `CombinedLoss.forward()` (`src/losses/combined.py:472`) 的公式：

```python
base_weight = max(0.3, 1.0 - total_extra_weight)
```

**分析**: 当前 P2 config 的 `total_extra_weight`:

| 方案 | boundary | aji | topology | size | contour | 总和 | base_weight |
|------|----------|-----|----------|------|---------|------|-------------|
| **当前 P2** | 0.5 | 0.2 | 0.1 | 0.1 | 0.1 | **1.0** | **0.3** (触底) |
| **Phase 1 提案** | 1.5 | 0.2 | 0.1 | 0.1 | 0.3 | **2.2** | **0.3** (触底) |

两者 `base_weight` 都落到地板 0.3。`boundary_weight` 从 0.5→1.5 不会降低 base_loss 占比（已经在 0.3），而是改变**辅助损失之间的相对排序**。

#### 12.1.3 PQ 早停已实现

`train.py:515-520` 已有完整的 PQ 早停逻辑：

```python
use_pq_early_stop = config['training'].get('use_pq_early_stop', False)
```

且 `validate()` 返回的 dict 已包含 `pq` 字段。只需在 config 设 `use_pq_early_stop: true` 即可启用。

### 12.2 Phase 1 实施方案（Config-Only）

**核心判断**: 不需要修改 `CombinedLoss` 或 `train.py` 代码。所有改动通过新建配置文件完成。

#### 12.2.1 新建配置文件

**文件**: `src/config/phase1_rebalance.yaml`

```yaml
# Phase 1: Loss Weight Rebalance + PQ Early Stop
# Based on: bf_instance_p2_20260205.yaml
# Changes: boundary 0.5→1.5, contour 0.1→0.3, PQ early stop ON

data:
  splits_dir: "data/splits"
  raw_data_dir: "data/raw/allen_segmented_fields_full"
  processed_data_dir: "data/processed"
  target_size: [1024, 1024]
  max_boxes_per_image: 30
  use_bf_only: true
  use_semantic_mapping: false

model:
  checkpoint: null
  freeze_encoder: true
  freeze_decoder: false
  use_adapter: false

training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.0001
  weight_decay: 0.0001
  warmup_epochs: 5
  early_stop_patience: 15
  use_pq_early_stop: true      # 改动: false → true

loss:
  type: "combined"
  pos_weight: 2.0               # 改动: 10.0 → 2.0 (文档清晰度,实际影响有限)
  boundary_weight: 1.5          # 改动: 0.5 → 1.5 (边界为辅助损失最高优先级)
  use_boundary: true
  use_aji: true
  aji_weight: 0.2
  box_expand: 0.1
  use_topology: true
  topology_weight: 0.1
  use_size: true
  size_weight: 0.1
  use_contour: true
  contour_weight: 0.3           # 改动: 0.1 → 0.3 (配合边界增强)

optimizer:
  type: "adamw"
  scheduler: "cosine_warmup"

output:
  checkpoint_dir: "checkpoints"
  save_every: 5
  experiment_name: "E_phase1_rebalance"
```

#### 12.2.2 新增 E2E Smoke Test

**文件**: `tools/smoke_test_e2e.py`

轻量运行时测试脚本（5个样本），用于补齐 Codex Phase 0 审核中指出的"源码模式匹配 + 缺少运行时验证"问题。

### 12.3 辅助损失相对权重效果

改动后的辅助损失占比变化：

| 损失 | 原权重 | 原占比 | 新权重 | 新占比 |
|------|--------|--------|--------|--------|
| base (Dice+BCE) | (auto) | 30% | (auto) | 30% |
| Boundary | 0.5 | 50% × 辅助 | **1.5** | **68%** × 辅助 |
| AJI | 0.2 | 20% × 辅助 | 0.2 | 9% × 辅助 |
| Topology | 0.1 | 10% × 辅助 | 0.1 | 5% × 辅助 |
| Size | 0.1 | 10% × 辅助 | 0.1 | 5% × 辅助 |
| Contour | 0.1 | 10% × 辅助 | **0.3** | **14%** × 辅助 |

边界相关损失 (Boundary + Contour) 从原来 60% → 改后 82% 的辅助损失份额。

### 12.4 请 Codex 审核

1. **`pos_weight` 动态限制效应是否准确？** 代码证据在 `src/losses/combined.py:442-443`
2. **`base_weight=0.3` 地板 + boundary 占辅助 68% 是否过激？** 是否需要调整地板
3. **PQ 早停是否应该无条件启用？** 还是作为可选项保留
4. **是否需要同步调整 `CombinedLoss` 的 `base_weight` 下限（0.3）？**
5. **Phase 1 仅 config-only 是否足够？** 还是需要代码级改动

### 12.5 与前文方案关系

- 本章属于第四节 §4.3 (参数调整) 的具体实施
- 不涉及 §4.1 的 L_neighbor / L_overlap（那是 Phase 2）
- PQ 早停对应 §3.1 中"切换到 PQ 或 Best-Match Dice"

---

## 十三、[Claude新增 | 2026-02-10] Phase 1 Oracle Smoke Baseline (v2)

> **标注说明**: 本章节由 Claude (Antigravity) 撰写，基于实际运行结果。
> **更新时间**: 2026-02-10 20:30 (v2: n=30, 双基线, TP/FP/FN)
> **运行命令**:
> - `conda run -n cellsam python tools/smoke_test_e2e.py --n_samples 30 --seed 42 --output smoke_pretrained_n30.csv`
> - `conda run -n cellsam python tools/smoke_test_e2e.py --n_samples 30 --seed 42 --checkpoint checkpoints/E29_bf_instance_p1_20260205_20260205_050953/best_model.pt --output smoke_finetuned_n30.csv`
>
> **注**: E29 = 第29个实验 (非 Epoch 29)。本次评估使用 GT boxes (Oracle 路线)，不涉及 DAPI 核检测框。
> E2E 路线使用 `src/detection/dapi.py` 的 DAPI 核检测方案生成框 (非 cellfinder)。

### 13.1 实验设置

| 项目 | 值 |
|------|-----|
| 推理入口 | `src/inference/core.py::segment_with_boxes` |
| 配置 | `InferenceConfig.default()` (threshold=0.5, argmax_prob, box_expand=0.1) |
| box 来源 | GT boxes (Oracle) |
| 设备 | CUDA |
| Conda 环境 | cellsam |
| **样本数** | **n=30** (seed=42, 从 71 个 val 样本中随机抽取) |
| **抽样方式** | `random.shuffle(val_ids)` with `seed=42`，固定可复现 |
| Per-sample CSV | `smoke_pretrained_n30.csv`, `smoke_finetuned_n30.csv` |

### 13.2 双基线对比

| 指标 | Pretrained (无微调) | E29 (实验#29) Finetuned | Δ |
|------|:---:|:---:|:---:|
| **BM-1to1 Dice** | 0.111 ± 0.021 | **0.593 ± 0.094** | **+0.482** |
| **BM-Coverage Dice** | 0.123 ± 0.028 | **0.601 ± 0.090** | **+0.478** |
| **Gap Dice** | 0.013 ± 0.031 | 0.008 ± 0.011 | −0.005 |
| **PQ@0.5** | 0.000 ± 0.000 | **0.326 ± 0.109** | **+0.326** |
| **SQ** | 0.000 ± 0.000 | 0.586 ± 0.030 | +0.586 |
| **RQ** | 0.000 ± 0.000 | 0.557 ± 0.189 | +0.557 |
| **TP** | 0.0 | **5.4 ± 2.9** | +5.4 |
| **FP** | 10.1 ± 4.9 | 4.4 ± 3.0 | −5.7 |
| **FN** | 10.5 ± 4.7 | 5.0 ± 3.1 | −5.5 |
| **AJI** | 0.045 ± 0.018 | **0.410 ± 0.127** | **+0.365** |
| **Semantic Dice** | 0.205 ± 0.047 | **0.720 ± 0.130** | **+0.515** |
| n_gt_cells | 10.5 ± 4.7 | 10.5 ± 4.7 | 0 (同一数据) |
| n_pred_cells | 10.1 ± 4.9 | 9.8 ± 4.6 | −0.3 |
| conflict_pixels | 42,103 ± 22,933 | 62,453 ± 50,315 | +20,350 |
| 耗时 | 1.3s/样本 | 1.3s/样本 | — |

### 13.3 TP/FP/FN 分析 (PQ=0 解释)

**Pretrained 模型**: TP=0, FP≈10, FN≈10 → 没有任何预测实例与 GT 实例达到 IoU≥0.5 匹配。所有预测实例都被判为 FP，所有 GT 实例都是 FN。这**不是实现 bug**，而是预训练模型对心肌细胞任务适应性极差的正常结果。

**E29 微调模型**: TP≈5.4, FP≈4.4, FN≈5.0 → 约 52% 的 GT 实例被正确匹配 (TP/n_gt ≈ 5.4/10.5)。FP 和 FN 都较高，说明分割质量和匹配率仍有提升空间。

### 13.4 关键发现

1. **微调效果显著**: BM-1to1 从 0.11 → 0.59 (+0.48)，PQ 从 0.00 → 0.33 (+0.33) (E29 = 第29个实验)
2. **PQ 仍远低于目标**: 当前 PQ=0.326 vs 目标 ≥0.42，差距 0.094
3. **BM-Dice 接近目标**: BM-1to1=0.593 vs 目标 ≥0.65，差距 0.057
4. **Gap Dice 很小 (0.008)**: 粘连/合并问题不严重（但 Codex 提醒：低 gap 可能因整体 dice 不够高而被掩盖）
5. **Conflict pixels 微调后反增**: pretrained 42K → finetuned 62K，说明微调后模型更自信但重叠区域更大
6. **SQ=0.586 偏低**: 匹配到的实例对平均 IoU 只有 0.586，边界精度有待提升

### 13.5 与 §5 验收标准对照

| 指标 | Pretrained | E29 微调 | §5 目标 | E29 差距 |
|------|:---:|:---:|:---:|:---:|
| Best-Match Dice | 0.111 | 0.593 | ≥0.65 | **−0.057** |
| PQ@0.5 | 0.000 | 0.326 | ≥0.42 | **−0.094** |
| IntrusionRate | 未测 | 未测 | <10% | — |
| ConflictRate | 4.0% | 5.9% | <5% | **+0.9%** |

> ConflictRate = conflict_pixels / (1024×1024) ≈ 62453/1048576 ≈ 5.96%

### 13.6 Phase 1 训练预期

Phase 1 配置变更 (boundary_weight 0.3→1.5, contour_weight 0.1→0.3, PQ early stop) 主要作用于：
- **边界精度 → 提升 SQ** (boundary_weight 5x 增加)
- **轮廓一致性 → 减少 FP/FN** (contour_weight 3x 增加)
- **PQ 早停 → 优化 TP 匹配率** (直接针对 RQ 优化)

结合 E29 基线 (PQ=0.326, SQ=0.586, RQ=0.557)，Phase 1 训练应优先提升 SQ 和 RQ。

### 13.7 请 Codex 审核

1. **双基线数据是否充分？** n=30, seed=42, 含 TP/FP/FN
2. **PQ=0 的 TP/FP/FN 解释是否消除了 bug 嫌疑？**
3. **E29 微调 checkpoint 是否是合适的 Phase 0 基线？** 还是应该用其他 checkpoint
4. **conflict_pixels 微调后增加是否需要关注？** 可能需要 Phase 2 的 L_overlap 来解决
5. **Phase 1 训练的预期收益方向是否合理？**

> **注**: 本章为 Oracle 路线基线（使用 GT boxes）；DAPI 框 E2E 基线需在独立小节报告。

---

## 十四、[Codex新增 | 2026-02-10] 第十三章(v2)审核结论与修订建议

> **标注说明**: 本章节由 Codex 基于已执行的轻量命令与回归测试结果给出审核意见。  
> **审核范围**: 第十三章双基线结论、`tools/smoke_test_e2e.py`、`src/metrics/instance_metrics.py`、`tools/test_unified_regression.py`。  
> **证据状态**: `python tools/test_unified_regression.py` 已通过 (`10 passed, 0 failed`)。

### 14.1 审核结论

本轮修改可判定为**技术上可用，结论方向基本可信**，但建议在正式归档前修复 2 个中风险文档问题（命名一致性与复现路径完整性）。

### 14.2 已确认有效的改动

1. 指标层已新增 PQ 细节实现：`compute_pq_detailed()` 已落地，并接入汇总指标。  
   - 代码位置：`src/metrics/instance_metrics.py` (`compute_pq_detailed`, `compute_all_metrics`)。
2. Smoke 脚本已支持可复现抽样与导出：`n=30`、`seed`、per-sample CSV、TP/FP/FN 输出。  
   - 代码位置：`tools/smoke_test_e2e.py` (`run_smoke_test`, `--seed`, `--output`)。
3. checkpoint 解包问题已修复：`load_cellsam_checkpoint` 三返回值在 smoke 脚本中正确接收。  
   - 代码位置：`tools/smoke_test_e2e.py`。
4. 统一回归测试可通过：`tools/test_unified_regression.py` 运行结果 10/10 通过。

### 14.3 发现的问题（按严重度）

1. **Medium**: 第十三章标题与实验路径表述有歧义。  
   - 章节标题写为 “E2E Smoke Test”，但正文明确本次为 **GT boxes (Oracle)**。  
   - 建议：将标题改为 “Oracle smoke baseline”，并将真正 DAPI 框 E2E 结果单列。
2. **Medium**: 命令中的 checkpoint 路径使用省略号，复现信息不完整。  
   - 示例：`checkpoints/E29_bf_.../best_model.pt`。  
   - 建议：写完整路径或唯一 checkpoint ID（含时间戳/实验目录）。
3. **Low**: 回归测试已扩展至 10 项，但部分文档描述仍写 “9-test”。  
   - 建议：统一为 “10-test regression suite”。
4. **Low**: 新增 `tp/fp/fn/sq/rq` 字段后，回归断言覆盖仍偏弱。  
   - 建议在 `tools/test_unified_regression.py` 中显式断言这些键存在，防止后续回退。

### 14.4 对第十三章关键结论的审核判断

1. “Pretrained 在该任务上 PQ=0” 的解释在当前证据下成立，且 TP/FP/FN 分解能够支持“非实现 bug”的判断。  
2. “E29 对 Pretrained 有明显提升”成立（BM-1to1、PQ、AJI、Semantic Dice 均显著提升）。  
3. “可进入 Phase 1 训练”判断合理，但应补充明确复现路径与命名一致性后再作为正式基线归档。

### 14.5 建议的最小修订清单（文档层）

1. 将第十三章标题改为：`Phase 1 Oracle Smoke Baseline (v2)`。  
2. 在运行命令中替换省略号路径为完整 checkpoint 路径。  
3. 将相关文档中的 `9-test` 统一改为 `10-test`。  
4. 在第十三章末尾增加一句：  
   - “本章为 Oracle 路线基线；DAPI 框 E2E 基线需在独立小节报告。”
