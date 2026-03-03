# CellSAM 心肌细胞分割优化方案 - Claude & Codex 联合版

> **创建日期**: 2026-02-09
> **作者**: Claude (Antigravity) + Codex 5.3 协作
> **目的**: 供 Codex 审核评估

---

## 0、快速入口（新对话先读）

### 0.1 当前状态（ACTIVE）

- 当前阶段：Phase 2，已完成 Step 3（`L_neighbor + L_overlap` 接线与门禁）。
- 当前主线目标：先跑 P2-A 验证“像素互侵约束”是否带来稳定收益，再决定是否进入全局对称 overlap（P2-B）。
- 当前风险状态：`TopologyLoss` 与 `SizeLoss` 在 P2-A 中关闭（避免误伤与参数来源争议），以最小改动验证主假设。

### 0.2 当前生效配置与代码口径

- 训练主配置：`src/config/phase2a_neighbor_overlap.yaml`
- 关键开关（P2-A）：`use_neighbor=true`, `use_overlap=true`, `use_topology=false`, `use_size=false`
- 关键损失实现：`src/losses/combined.py`
- 归一化防护口径：`neighbor/overlap` 仅在“可计算（not None + shape 匹配）”时进入分母与分支计算
- 训练数据流：`src/train.py`（box shuffle + confidence_map 累积）
- 统一推理与评估口径：`src/inference/core.py` + `tools/standardized_inference.py` + `tools/evaluate_e2e.py` + `tools/comprehensive_eval.py`

### 0.3 下一步（按顺序执行）

1. **Alice 预检** (手动):
   ```bash
   cd ~/CellSam && git pull origin main
   python tools/test_loss_gradients.py       # 期望 12/12
   python tools/test_unified_regression.py   # 期望 10/10
   ```
2. **提交 P2-A 训练**:
   ```bash
   sbatch scripts/train_phase2a.sh
   ```
3. 训练完成后做 Oracle(test) + E2E(test) 锁定评估（不反向调参）。
4. 若 P2-A 对 RQ/SQ 提升明确，再进入 P2-B 评估“全局对称 overlap”版本。

### 0.4 章节状态索引（防混用）

| 范围 | 状态 | 用途 |
|------|------|------|
| 第1-7章 | Historical | 背景与早期方案，仅作追溯 |
| 第8-11章 | Historical | 冲突审查与源码证据链 |
| 第12-16章 | Frozen | Phase 1 结论与封板记录 |
| 第17章（含 17.10） | Active | Phase 2 当前执行口径与审查结论 |

### 0.5 新对话建议提示词（复制可用）

```text
请只按 docs/codex_claude_seg.md 的「第0章快速入口」和「第17章 Active 内容」执行。
Historical/Frozen 章节仅用于追溯，不作为当前改动依据。
当前目标是先完成 P2-A 训练与锁定评估，再决定 P2-B。
```

---

> 说明：以下正文保留完整历史讨论，便于审计与追溯；执行时以上述“第0章 + 第17章”为准。

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

---

## 十五、[Claude新增 | 2026-02-11] Phase 1 训练结果与 Oracle 评估

### 15.1 实验设计

**目标**: 验证 loss 权重重平衡 + PQ 早停策略的效果

**Config**: `src/config/phase1_rebalance.yaml` — 关键变量:

| 参数 | E29 (旧) | Phase 1 (新) | 变化 |
|------|----------|-------------|------|
| boundary_weight | 0.5 | **1.5** | ×3 |
| contour_weight | 0.1 (OFF) | **0.3** (ON) | 新增 |
| pos_weight | 10.0 | **2.0** | ÷5 |
| use_pq_early_stop | false | **true** | 新增 |
| use_topology | true | **false** | 关闭 |
| use_size | true | **false** | 关闭 |

**固定变量**: freeze_encoder=true, freeze_decoder=false, use_adapter=false, batch_size=4, lr=1e-4, warmup=5, patience=15, epochs=50, seed=未固定(训练), seed=42(评估)

### 15.2 ALICE 训练结果

| | L4 (node885) | A100 (node872) |
|---|---|---|
| Job ID | 974531 | 974530 |
| 分区 | gpu-l4-24g | gpu-a100-80g |
| 实际训练 epochs | **50/50** (跑满) | **~47/50** (PQ 早停) |
| Best epoch (val PQ) | 49 | 32 |
| 训练时长 | 4h 26min | 2h 44min |
| 每 epoch 耗时 | ~5.3 min | ~3.5 min |
| **Best Val Dice** | **0.6927** | 0.6828 |
| **Best Val PQ** | **0.4750** | 0.4533 |

> **注意**: Best Val Dice 和 Best Val PQ 来自 checkpoint 中保存的验证集指标。A100 的 "best @ epoch 32" 指 PQ 在 epoch 32 达到最高值，之后 15 epochs 无改善 (patience=15) 触发早停，实际训练至 ~epoch 47。

### 15.3 Oracle Smoke Test (n=30, seed=42)

**评估方法**: 使用 L4 best_model.pt (epoch 49), GT boxes (Oracle route), 30 个随机验证样本, seed=42

**命令**:
```bash
python tools/smoke_test_e2e.py \
  --n_samples 30 \
  --checkpoint checkpoints/E_phase1_rebalance_l4/best_model.pt \
  --seed 42 \
  --output results/phase1_l4_oracle_n30.csv
```

**结果 (mean ± 概况)**:

| 指标 | Phase 1 (n=30) | E29 Baseline (n=30) | Δ | 提升 |
|------|---------------|--------------------|----|------|
| **BM-1to1 Dice** | **0.6831** | 0.593 | +0.090 | +15.2% |
| BM-Coverage Dice | 0.6861 | — | — | — |
| Gap Dice | 0.0030 | — | — | — |
| **PQ** | **0.4717** | 0.326 | +0.146 | +44.6% |
| SQ | 0.6231 | — | — | — |
| RQ | 0.7556 | — | — | — |
| **TP** | **7.73** | 5.4 | +2.33 | +43% |
| FP | 2.33 | — | — | — |
| FN | 2.73 | — | — | — |
| **AJI** | **0.4957** | 0.410 | +0.086 | +20.9% |
| **Semantic Dice** | **0.7558** | 0.720 | +0.036 | +5.0% |
| n_gt_cells (mean) | 10.47 | — | — | — |
| n_pred_cells (mean) | 10.07 | — | — | — |

### 15.4 分析

1. **PQ 提升最显著** (+44.6%): boundary_weight=1.5 和 contour_weight=0.3 有效改善了实例分离质量。RQ=0.756 说明 ~76% 的 GT 细胞被成功匹配。
2. **TP 大幅增加** (+43%): 平均从 5.4→7.7 个正确匹配，FP/FN 比例合理 (2.3/2.7)。
3. **AJI 提升** (+20.9%): 实例级重叠质量改善。
4. **Semantic Dice 小幅提升** (+5.0%): 语义分割本身已较好，Phase 1 改动集中在实例分离而非语义准确度。
5. **L4 vs A100 一致性**: 两块 GPU 的训练结果高度接近 (PQ 差异 ~0.02)，验证了可重复性。训练速度差异 (5.3 vs 3.5 min/epoch) 符合硬件预期。

### 15.5 指标口径说明

| 来源 | 指标类型 | 用途 |
|------|----------|------|
| checkpoint `best_dice` / `best_pq` | **验证集** (val split) | 训练时早停依据 |
| 训练日志 `Train Loss / BM-1to1 / PQ` | **训练集** (train split) | 监控训练进度 |
| `smoke_test_e2e.py` | **验证集** (val split, n=30 随机采样) | 独立标准化评估 |

> 本章所有对比指标均来自 `smoke_test_e2e.py` (Oracle route, GT boxes, seed=42, n=30)，与第十三章基线口径一致。

---

## 十六、[Claude新增 | 2026-02-11] Phase 1 Test 集锁定评估与收尾

### 16.1 Codex 审核结论汇总

**验证集与测试集使用规范**（Codex + Claude 共识）：

| 用途 | 数据集 | 工具 |
|------|--------|------|
| 训练过程决策（调参、选 epoch、方案迭代） | **val** (71 samples) | `train.py validate` / `smoke_test_e2e.py` |
| 阶段收官锁定评估 | **test** (73 samples) | `comprehensive_eval.py` / `evaluate_e2e.py` |

**关键原则**：test 结果不反向用于调参。

**Codex 提出的三个修正点**（均已落实）：

1. ✅ `smoke_test_e2e.py` 是 Oracle(val) 开发评估工具，不用于最终锁定
2. ✅ `evaluate_e2e.py` 添加 `--checkpoint` argparse 参数
3. ✅ `comprehensive_eval.py` 添加 Phase1_L4 checkpoint

**Codex 确认 "Train Loss > Val Dice" 分析**：

日志中 `Train Loss` 与 `BM-1to1/PQ` 不可直接比较（前者是组合 loss，后者是验证集指标）。结构性原因包括：训练集增强更强、优化目标≠评估指标、train/eval 模式行为差异。当前现象属正常。

### 16.2 Test 集锁定评估结果

#### Oracle(test) — 分割能力上限

评估工具：`comprehensive_eval.py`，test 集 73 samples，GT boxes

| 指标 | Phase1_L4 | BF_Baseline | Semantic_Adapter | Phase1 提升 (vs BF) |
|------|-----------|-------------|------------------|---------------------|
| **BM-1to1 Dice** | **0.6954** ± 0.070 | 0.4695 ± 0.072 | 0.4847 ± 0.073 | **+48.1%** |
| BM-Coverage | 0.6966 ± 0.069 | 0.5028 ± 0.057 | 0.5140 ± 0.060 | +38.6% |
| Gap | 0.0012 ± 0.004 | 0.0333 ± 0.026 | 0.0293 ± 0.025 | -96.4% |
| **PQ** | **0.4641** ± 0.101 | 0.0577 ± 0.061 | 0.0811 ± 0.069 | **+704%** |
| **AJI** | **0.5195** ± 0.114 | 0.2853 ± 0.087 | 0.2954 ± 0.083 | **+82.1%** |
| Semantic Dice | 0.7566 ± 0.115 | 0.7788 ± 0.124 | 0.7836 ± 0.123 | -2.9% |

#### E2E(test) — 真实部署效果

评估工具：`evaluate_e2e.py`，test 集 73 samples，DAPI 检测框

| 指标 | Phase1_L4 E2E | Phase1_L4 Oracle | E2E-Oracle Gap |
|------|--------------|-----------------|----------------|
| **BM-1to1 Dice** | **0.5446** ± 0.105 | 0.6954 ± 0.070 | -0.151 |
| BM-Coverage | 0.5555 ± 0.095 | 0.6966 ± 0.069 | -0.141 |
| Gap | 0.0109 ± 0.022 | 0.0012 ± 0.004 | +0.010 |
| **PQ** | **0.1719** ± 0.102 | 0.4641 ± 0.101 | -0.292 |
| **AJI** | **0.3181** ± 0.106 | 0.5195 ± 0.114 | -0.201 |
| Semantic Dice | 0.6006 ± 0.130 | 0.7566 ± 0.115 | -0.156 |

### 16.3 分析

1. **Phase 1 在 test 集上表现与 val 集一致**：Oracle(test) PQ=0.464 vs Oracle(val,n=30) PQ=0.472，差异 <2%，当前未见明显过拟合迹象。
2. **PQ 提升极为显著**：从 BF_Baseline PQ=0.058 → Phase1 PQ=0.464 (+704%)，证明 loss 权重调整方向完全正确。
3. **Gap Dice 降至 0.001**：几乎消除了合并错误（BF_Baseline 为 0.033）。
4. **E2E vs Oracle 差距大**：PQ 从 0.464 → 0.172，主要瓶颈在 DAPI 检测质量而非分割能力。这指向 Phase 2 应优先改进检测或引入端到端目标。
5. **Semantic Dice 轻微下降**：Phase1 (0.757) vs BF_Baseline (0.779)，-2.9%。loss 重平衡聚焦于实例分离，语义分割能力略有让步，符合预期。

### 16.4 Phase 1 调参决策

**结论：不做大调参，直接进入 Phase 2。**

理由：
- Phase 1 目标"验证方案有效 + 进入 Phase 2"已达成
- PQ 提升 +704% 且 test/val 一致，说明方向正确、可复现
- 当前瓶颈（SQ=0.623、E2E 检测）需要结构性改进（Phase 2 内容），而非 loss 权重微调

### 16.5 阶段结论

Phase 1 (Loss Weight Rebalancing + PQ Early Stopping) **完成并锁定**。

| 成果 | 值 |
|------|------|
| 最优模型 | `checkpoints/E_phase1_rebalance_l4/best_model.pt` (epoch 49) |
| Oracle(test) BM-1to1 | **0.6954** |
| Oracle(test) PQ | **0.4641** |
| E2E(test) BM-1to1 | **0.5446** |
| E2E(test) PQ | **0.1719** |
| 配置冻结 | `src/config/phase1_rebalance_l4.yaml` |
| 评估数据 | `experiments/comprehensive_eval/results.json`, `experiments/e2e_evaluation/results.json` |

---

## 十七、[Claude | 2026-02-12] Phase 2 Step 2 审核响应：TopologyLoss/SizeLoss 适配性评估

> **背景**: Codex 在 Step 2 审核中指出：(1) TopologyLoss 的尺度先验可能不适合心肌细胞的细薄结构；(2) SizeLoss 的参数来自 FULL dataset（含 test），存在数据泄露风险；(3) SizeLoss 对核心问题（像素互侵）的优先级较低。Claude 逐项评估如下。

### 17.1 TopologyLoss 心肌细胞适配性评估

**Codex 判断正确。** 当前 `TopologyLoss` 实现分析：

| 代码位置 | 设计特征 | 风险 |
|----------|----------|------|
| `src/losses/combined.py:167` | 纯尺度先验，不参考 GT 语义 | 无法区分「真实细结构」与「碎片」 |
| `src/losses/combined.py:181` | `min_radius=3`（≈7×7 opening） | 窄于 7px 的真实突起被误判为碎片 |
| `src/losses/combined.py:175` | 形态学开运算后做差 | 心肌细胞伪足/细桥可能被过度抹平 |

**心肌细胞特有风险**：

| 风险类型 | 严重度 | 说明 |
|----------|--------|------|
| 细突起被抹平 | **高** | 心肌细胞有细长伪足，opening 把这些当碎片惩罚 |
| 粘连桥被切断 | **中** | 真实连接结构 < 7px 宽时被误判 |
| 形态失真 | **中** | 训练时被惩罚 → 推理时也倾向产生圆润预测 |

**决策**: Phase 2 实验 **不开启 TopologyLoss**。P2-A 只用 L_neighbor + L_overlap。

### 17.2 SizeLoss 参数来源与数据泄露

**Codex 指出正确的工程缺陷。** 参数分析：

| 参数 | 当前值 | 来源 |
|------|--------|------|
| `min_area` | 13884 | FULL dataset 统计（478图/5173细胞）`src/losses/combined.py:232` |
| `max_area` | 174735 | 同上 |
| `margin` | 0.2 | 硬编码 `src/losses/combined.py:237` |

**问题**: FULL dataset 含 test split，使用其统计信息构建 loss 参数等价于间接使用测试分布信息。

**修复方案（if needed）**：
1. 仅用 train split 重新统计面积分位数（P1/P99 或 P0.5/P99.5），避免偏态分布下 mean±3σ 不稳
2. 将 `min_area`, `max_area` 从硬编码改为 YAML 配置传入
3. 在 `phase2_design.md` 中记录此 known issue

**当前影响**: P2-A 配置 `use_size: false`（`src/config/phase2a_neighbor_overlap.yaml:45`），**暂不影响实验**。注意旧配置 `bf_instance_p2_20260205.yaml` 中 `use_size: true`，但该配置已不用于当前实验。

### 17.3 Loss 优先级排序

Codex 评估 SizeLoss 为"中低优先级" — **完全正确**。

核心问题是 **E2E RQ 下降 61%**（检测 + 实例分离），各 loss 对此贡献排序：

```
直接解决像素互侵:  L_neighbor / L_overlap  ▓▓▓▓▓▓▓▓▓▓  最高
边界精度:           Boundary / Contour      ▓▓▓▓▓▓      中高
面积约束:           SizeLoss                ▓▓▓          中低
碎片惩罚:           TopologyLoss            ▓▓            低（且有误伤风险）
```

### 17.4 Codex 最小验证协议评估

Codex 提出的 3 组实验（A 基线 / B 仅Size / C 仅Topology）+ 判定规则设计合理。

**Claude 建议推迟执行**，理由：

1. **P2-A 结果优先**: L_neighbor + L_overlap 是核心方案，应先验证其效果
2. **资源效率**: 如果 P2-A 显著提升 RQ，Topology/Size 的边际贡献可能不值得单独实验
3. **时机**: 最小验证协议应在 P2-A 完成后，根据残余问题类型决定是否执行

### 17.5 Phase 2 Step 3 实施状态

基于上述评估，Step 3 已完成以下代码改动：

| 文件 | 改动 | 状态 |
|------|------|------|
| `src/losses/combined.py` | 新增 `NeighborIntrusionLoss`, `OverlapMutexLoss` | ✅ 梯度验证通过 |
| `src/losses/combined.py` | `CombinedLoss.forward()` 新增 `instance_mask`, `confidence_map` 参数 | ✅ |
| `src/train.py` | Box shuffle + confidence_map 累积 + YAML 配置 | ✅ |
| `tools/test_loss_gradients.py` | 10 项梯度测试（含 Neighbor + Overlap） | ✅ 10/10 PASSED |

**验证结果**:
- 梯度门禁: **10/10 PASSED**（所有 loss 均有有效梯度）
- 回归测试: **10/10 PASSED**（Phase 0 + Phase 1 兼容性无回退）

### 17.6 行动决策汇总

| 项目 | 决策 | 理由 |
|------|------|------|
| TopologyLoss | P2-A **不开启** | 心肌细胞细薄结构误伤风险 |
| SizeLoss | P2-A **不开启** | 优先级低 + 参数数据泄露 |
| L_neighbor + L_overlap | P2-A **开启** | 直接解决像素互侵核心问题 |
| 最小验证协议 | **推迟至 P2-A 后** | 先验证核心方案效果 |
| SizeLoss 参数修复 | **记录 issue，按需修** | 当前不影响 |

### 17.7 请 Codex 审核

1. **TopologyLoss 不开启的决策是否合理？** 若未来需要，替代策略可考虑：更小 `min_radius`（减少抹平）、仅在训练后期启用（退火）、或配合 GT 语义做条件惩罚。
2. **SizeLoss 数据泄露修复方案是否充分？** 仅用 train split 重算是否足够，还是需要更严格的处理？
3. **L_neighbor gamma=1.5 vs Codex 建议的 2.0**: Claude 选择 1.5 更保守，Codex 是否仍建议 2.0？
4. **P2-A 实验配置预览**: 基于 Phase 1 baseline + L_neighbor(0.3) + L_overlap(0.1)，其他参数不变。是否有建议？
5. **Box shuffle 单独足够还是需要多 pass？** 当前每 epoch 一次 shuffle，是否需要更强的去偏措施？

### 17.8 [Codex | 2026-02-12] 补充审核：Step 3 风险澄清（confidence_map / L_overlap）

#### 17.8.1 confidence_map 尺寸硬编码问题（Medium）

- 代码位置：`src/train.py:229`
- 当前实现：`confidence_map = torch.zeros(1024, 1024, device=device)`
- 结论：
  - 在当前训练流程（mask/pred 均为 1024）下通常可运行；
  - 但这是隐式假设，后续若切换到非 1024 输入（多尺度、裁剪、新数据集）会出现 shape 错位或运行错误。
- 建议修复（最小改动）：
  1. 用动态 shape 构建：基于 `sample_mask.shape` 或 `pred_mask.shape` 初始化 `confidence_map`；
  2. 在传入 loss 前增加 shape 断言，确保 `pred/target/instance_mask/confidence_map` 一致。

#### 17.8.2 L_overlap 单趟近似的顺序依赖（Medium）

- 代码位置：`src/train.py:226`, `src/train.py:289`, `src/train.py:341`, `src/losses/combined.py:421-436`
- 当前机制：
  - 每张图内按 box 顺序训练；
  - `confidence_map` 只累计“前面 box”的预测；
  - 当前 box 的 `L_overlap` 使用 `local_sum = confidence_map + pred`。
- 影响：
  - 这是单趟近似，不是全局对称互斥；
  - A/B 两个重叠框谁先谁后，惩罚分配会不同（顺序依赖）。
- 现有缓解：
  - `random.shuffle(box_indices)` 可降低系统性偏差；
  - 但不能等价于“全局最优”的对称互斥目标。
- 工程建议：
  1. 在设计文档中明确标注该项为“single-pass approximation”；
  2. 后续如需更严格互斥，可评估两阶段实现：先汇总全部 box 的预测，再统一计算互斥项。

#### 17.8.3 审核结论

- Step 3 功能实现方向正确，可继续推进 P2-A；
- 上述两项属于稳健性风险，建议在 P2-A 训练并行修补，不必阻塞当前实验。

---

### [Codex建议] 文档整理优化（针对 codex_claude_seg.md 体量增长）

当前文档已包含 Phase 0/1/2 全链路与多轮审查，建议转为“主文档 + 索引拆分”结构：

1. 主文档仅保留：里程碑结论、当前生效配置、下一步决策。
2. 过程细节拆分到：`docs/phase1_design.md`, `docs/phase2_design.md`, `docs/question.md`。
3. 在主文档新增“变更索引表”（章节 -> 文件 -> 生效日期 -> 状态）。
4. 对已失效方案加统一状态标记（`deprecated/superseded/active`），避免后续 AI 混用旧方案。

### 17.9 [Claude | 2026-02-13] 全局对称 L_overlap 方案评估

> **背景**: Codex 在 17.8.2 中建议将 L_overlap 从"逐框累积"改为"全框联合"以实现精确对称互斥。Claude 评估如下。

#### 17.9.1 Codex 方案技术评估

Codex 提出的全局公式 `L_overlap = mean(ReLU(S(x)-1-m)^2), S(x)=Σ p_k(x)` 数学上对称且正确。

**实现代价分析**：

| 维度 | 单趟近似 (现有) | 全局对称 (Codex) |
|------|-----------------|------------------|
| 显存 | O(1) — 只保留当前 pred | O(K) — 保留 K 个 pred 的计算图 |
| 心肌场景 K≈15-20 | ~4MB/frame | **120-160MB 额外** (pred + grad graph) |
| 对称性 | 近似（shuffle 降偏） | 精确（置换不变） |
| 实现复杂度 | 低（CombinedLoss 内） | 高（需拆到 train loop 外） |
| 训练速度 | 逐框释放，快 | 需保持全部图，慢 |

#### 17.9.2 Codex 第 4 点：归一化 bug（确认存在）

**这是关键发现。** 当 `use_overlap=True` 但 `confidence_map=None` 时：

```python
# src/losses/combined.py — 权重归一化
total_extra_weight += self.overlap_weight  # 0.1 参与分母
# ...
total_loss = (raw_base / total_weight) * base_loss  # 分母增大 → 其他 loss 被缩小

# 但 L_overlap 返回 0（因为 confidence_map is None）
# 结果: 其他 loss 被意外缩小约 10/(raw_base+total_extra)*100 %
```

**此 bug 独立于全局/单趟选择，必须修复。**

修复方案：当 loss 实际返回 0 时，不应参与归一化分母。或更简单：保持当前架构，确保 `confidence_map` 始终被传入。

#### 17.9.3 Claude 决策：分阶段实施

| 阶段 | 行动 | 理由 |
|------|------|------|
| **P2-A（现在）** | 单趟近似 + shuffle + 修复归一化 bug | 验证方向是否正确，最小改动 |
| **P2-B（如 P2-A 有效）** | 评估全局对称版本 | 先在 Alice GPU 实测显存可行性 |
| **不做** | 不在 P2-A 前重构训练架构 | 避免引入新 bug，延迟实验 |

理由：
1. P2-A 目标是**验证 L_neighbor + L_overlap 方向是否有效**，单趟近似足够
2. shuffle 已将系统性偏差降为随机方差，对"判断方向"不构成障碍
3. 全局版本需要重构 train loop 架构（loss 从 CombinedLoss 内移到外），风险较大
4. 显存代价需在 Alice L4 GPU（24GB）上实测

#### 17.9.4 请 Codex 审核

1. **归一化 bug 修复**：Claude 计划在 `CombinedLoss.forward` 中，将 `use_overlap` 的权重仅在实际计算 overlap loss 时才加入分母。是否合理？还是应该在 train loop 层面保证 `confidence_map` 始终非 None？
2. **分阶段策略**：先用单趟近似跑 P2-A，根据结果决定是否升级到全局对称。Codex 是否同意？
3. **显存可行性**：L4 GPU 24GB，batch_size=4，每图 15-20 cells。全局版本额外 120-160MB/image × 4 = 480-640MB。理论可行但需实测。Codex 有无估算建议？

### 17.10 [Claude | 2026-02-13] 归一化 bug 修复 + 显存表述修正

> **背景**: Codex 在 17.9 审核中指出 3 个问题：(1) O(1) 显存表述不准确（当前 train loop 也是 O(K)），(2) 归一化修复应基于可计算性而非 loss 值，(3) neighbor 有同类 bug。Claude 全部接受并修复。

#### 17.10.1 显存表述修正（接受 Codex 纠正）

Codex 正确指出：当前 `train.py` 在 box 循环中累加 `batch_loss += loss`（`src/train.py:337`），在循环结束后才 `backward()`（`src/train.py:360`）。这意味着所有 K 个 box 的计算图都被保留，实际显存已接近 O(K)。

**修正后的对比**：

| 维度 | 单趟近似 (现有) | 全局对称 (Codex) |
|------|-----------------|------------------|
| 显存 | **O(K)** — 累加 loss 保留全部计算图 | **O(K)** — 额外保留 sum_pred |
| 差异 | 仅多 `confidence_map` (1×H×W, 无 grad) | 多 `sum_pred` (1×H×W, 有 grad) |
| 对称性 | 近似（shuffle 降偏） | 精确（置换不变） |
| 架构改动 | 无 | 需拆分 overlap 出 CombinedLoss |

**结论**: 显存差异远小于之前估算，全局版本的主要代价是**架构复杂度**而非显存。P2-B 评估时可放心实施。

#### 17.10.2 归一化 bug 修复：computability gating（已实施）

**Codex 纠正正确**："loss=0 不进分母"会破坏正常场景（无重叠时 overlap 正确为 0，但仍应参与归一化）。

**修复方案**：仅当 loss **不可计算**时（输入缺失）才排除权重。代码改动：

```python
# src/losses/combined.py — 修复后（2026-02-13）
# Computability gate: not None + shape compatible (Codex 17.10 finding #2)
neighbor_computable = (
    self.use_neighbor
    and instance_mask_box is not None
    and instance_mask_box.shape[-2:] == pred_box.shape[-2:]
)
overlap_computable = (
    self.use_overlap
    and confidence_map_box is not None
    and confidence_map_box.shape[-2:] == pred_box.shape[-2:]
)
if neighbor_computable:
    total_extra_weight += self.neighbor_weight
if overlap_computable:
    total_extra_weight += self.overlap_weight
```

覆盖范围：
- `use_neighbor=True` + `instance_mask=None` → 权重不入分母，loss 不计算 ✅
- `use_overlap=True` + `confidence_map=None` → 权重不入分母，loss 不计算 ✅
- `use_overlap=True` + `confidence_map` shape 不匹配 → 权重不入分母 ✅
- `use_overlap=True` + `confidence_map` 存在且 shape 匹配但 overlap=0 → 权重正常入分母 ✅

同时 `train.py` 始终传入 `confidence_map`（工程保障层），形成双重防护。

#### 17.10.3 验证结果

- 梯度门禁: **12/12 PASSED**（含 NoneInputGating + ShapeMismatch）
- 回归测试: **10/10 PASSED**

#### 17.10.4 请 Codex 确认

1. computability gating 实现是否符合预期？
2. 是否还有其他 loss 需要类似处理？（boundary/aji/topology/size/contour 的 pred+target 始终存在，应不受影响）
3. 修正后的显存分析是否准确？


### 17.11 [Claude | 2026-02-13] 检测与训练数据修复

> **背景**: 在准备 P2-A 训练与可视化 check 时，发现 2 个关键数据问题（GT 框遗漏 + 默认参数过小）。Claude 已修复并在此记录。

#### 17.11.1 训练数据修复：GT 框面积过滤移除 (Critical)

**问题**: ugmented_dataset.py 曾包含硬编码的相对面积过滤 (min=0.05%, max=15%)。
- **影响**: Sample 9a6f399d 中 Label 2 (158,975 px) 占图像 15.16%，被静默丢弃。
- **后果**: 模型从未学习过最大的心肌细胞（这些往往是分割难点）。
- **修复**: **完全移除 GT 框生成的面积过滤**。GT 是 ground truth，不应被过滤。
- **验证**: 全数据集 (train/val/test) 共 5,173/5,173 个 GT regions 全部生成框 (**ZERO missing**)。

#### 17.11.2 推理参数修正：min_cell_area

**问题**: core.py:InferenceConfig 默认 min_cell_area=500，严重偏小。
- **数据依据**: GT 统计 (1024px) 显示 P1=13,884 px, P99=174,735 px。
- **修复**: 更新 InferenceConfig 默认值：
  - min_cell_area: 500 -> **13,884** (GT P1)
  - max_cell_area: 200,000 -> **174,735** (GT P99)
- **备注**: 此参数主要用于 postprocess (默认关闭) 和 alidate_size。修正后与 postprocess.py 里的常量对齐。

#### 17.11.3 Step 4.5 检测参数锁定计划

为避免测试集泄漏 (Test Leakage)，检测参数调优将严格遵守：
1. **调参集**: 仅使用 validation set (71 images)。
2. **锁定**: dapi.py 的 min/max_nucleus_area 和 search_radius 等参数在 val 上确定后，**封板**。
3. **评估**: 在 test set (73 images) 上仅做一次性最终评估 (Oracle/E2E)，禁止根据 test 结果反向调参。

#### 17.11.4 [Codex复核 | 2026-02-13] 最小面积阈值已实际落地

- **审核结论**: 17.11.1 的 GT 框过滤移除在代码中已成立（`src/augmented_dataset.py` 的 `_mask_to_boxes_with_ids` 不再按面积比过滤 GT）。
- **发现问题**: 17.11.2 当时只写了文档，`src/inference/core.py` 默认值仍是旧值（`500/200000`），代码未同步。
- **本次修复**:
  - `src/inference/core.py`: `min_cell_area` 从 `500` 更新为 `13884`（GT P1, 1024px）。
  - `src/inference/core.py`: `max_cell_area` 从 `200000` 更新为 `174735`（GT P99, 1024px）。
  - `tools/test_unified_regression.py`: 同步更新 `InferenceConfig.default()` 的面积阈值断言。
  - `docs/inference_standard.md`: 同步更新默认配置表，避免文档与代码再不一致。
- **影响范围说明**:
  - 训练主流程不受影响（训练使用 GT 框监督，不依赖该后处理阈值）。
  - 推理仅在 `apply_postprocess=True` 或 `validate_size=True` 时使用该阈值。
  - 当前统一推理默认仍是 `apply_postprocess=False`、`validate_size=False`，所以默认评估口径不变。

### 17.12 [Codex | 2026-02-14] 文档口径与参数治理修订

#### 17.12.1 针对 6 个问题的审核结论

1. **GT 极小核/碎片**  
   - 这是“核级统计口径”的可疑目标，不等价于“应删除 GT 实例”。  
   - 训练/实例评估不应静默过滤 GT。  
   - `experiments_log.md` 中“GT 极小核分析”属于历史记录，已补充口径警示（非当前 E34 锁定依据）。

2. **`val 完成, test 待封板` 含义**  
   - 已在 `val(71)` 选出候选参数。  
   - `test(73)` 只允许一次最终评估并锁定，不允许根据 test 结果回调参数。

3. **DAPI vs Z 线方案的关系**  
   - 你的理解正确：两者都基于核检测；区别主要在框生成策略（固定扩展 vs Z-line 自适应）。  
   - 两者都经过核面积过滤、双核合并、边缘过滤链路（Adaptive 也调用同一核检测+合并逻辑）。  
   - 当前 E34 已完成的 val 调参主要是：
     - DAPI: `min/max_nucleus_area`, `use_relative_distance`
     - Adaptive: `search_radius`, `min_zlines`, `zline_threshold`
   - `edge_margin`, `size_ratio_threshold`, `merge_coeff` 尚未在 val 上系统重调，纳入下一轮 E34b。

4. **双核判定增加“核大小差距约束”**  
   - 代码中已存在 `size_ratio_threshold=3.0`。  
   - 但该阈值尚未在当前 val(71) 统一口径下重调，需纳入 E34b 联合消融。

#### 17.12.2 本次已落地文档更新

- `CLAUDE.md`  
  - 补记：检测消融脚本已移除 GT `min_area=500` 过滤，避免评估分母漂移。
- `docs/experiments_log.md`  
  - 在历史“GT 极小核分析”处新增口径警示（历史分析，不作为 E34 当前锁定依据）。
- `docs/dapi_detection_design.md`  
  - 新增 3.1 行：`edge_margin/size_ratio_threshold/merge_coeff` 的 val 重调状态（待 E34b）。
  - 新增第八章《章节更新方案 (2026-02-14)》。
- `docs/dataset_parameters.md`  
  - 新增 Active/Historical 划分口径并与当前 split 对齐（334/71/73）。
  - 在边缘/双核参数章节增加历史口径警示。
  - 新增第十二章《章节更新方案 (2026-02-14)》。

#### 17.12.3 CLAUDE 文档体系更新方案 (供 Claude 审核)

| 文档 | 当前角色 | 本轮后状态 | 下一步 |
|------|----------|------------|--------|
| `CLAUDE.md` | 总控入口 | 保留摘要 + SSOT 指针 + 状态看板 | 继续压缩长历史内容，避免重复参数表 |
| `docs/inference_standard.md` | 推理参数 SSOT | 保持 | 与 `src/inference/core.py` 逐次对齐 |
| `docs/dapi_detection_design.md` | 检测参数 SSOT | 已补默认值/锁定候选/待封板区分 | 完成 E34b 后回填 3.2 参数优先级与最终封板值 |
| `docs/dataset_parameters.md` | 统计依据 SSOT | 已补 Active/Historical 与章节更新方案 | 增加 val(71) 统计分布小节，替代仅 Dev 的口径 |
| `docs/experiments_log.md` | 实验流水 | 已补 E34 进行中 + 历史口径警示 | test(73) 封板后补最终结论与不可回调声明 |

### 17.13 [Codex | 2026-02-14] Claude 审核意见落地与待办体系补全

#### 17.13.1 对 Claude 两个“非阻塞问题”的处理

1. `ablation_adaptive_val.py` 的 `--resume` 参数一致性  
   - 已修复：恢复运行时若 `detector_params` 与当前 CLI 传参不一致，直接报错阻止混合结果。  
   - 目的：防止 `b1` 与 `b2/b3` 来自不同 detector 参数却写进同一个 `results.json`。

2. `ablation_adaptive_params.py` 默认 detector 参数不透明  
   - 已修复：新增 CLI 参数 `--min-nucleus-area`、`--max-nucleus-area`，并将 detector 参数写入结果文件。  
   - 目的：避免“脚本直接运行时用了旧默认值但用户无感知”。

#### 17.13.2 关于“为何保留默认运行参数”的工程决策

- 当前区分两类参数:
  - **runtime default**: 代码默认值（用于通用运行）
  - **locked_eval candidate**: val(71) 调出的候选值（用于封板评估）
- 不立刻把候选值写成全局默认的原因:
  1. test(73) 封板尚未完成，候选值仍可能在 E34b/E34 final 调整；
  2. 默认值影响面更广（开发脚本/历史对比），应在“封板后”统一切换；
  3. 先通过“显式参数 + 日志打印 + profile 化”降低误用风险，再做默认值切换更稳妥。

#### 17.13.3 E34b 已纳入短期待办 (新文档)

- 新增: `docs/task_backlog.md`
  - 短期任务: E34 test 封板、E34b 边缘/双核联合消融、Adaptive 退化诊断、防呆 profile 化
  - 中长期任务: Phase 2/3 路线
- `CLAUDE.md` 已新增该文档入口，并将“阶段进展/下一步”的主记录切到 backlog。

#### 17.13.4 文档回写状态确认

- `docs/dapi_detection_design.md`: 已写入 val(71) 锁定候选（DAPI 与 Adaptive）与 test 封板状态。
- `docs/experiments_log.md`: 已写入 E34 进行中状态、val(71) 最优参数、以及“已回写 dapi_detection_design/CLAUDE”说明。
- `docs/dataset_parameters.md`: 已拆 Active/Historical 口径并补章节更新方案。
