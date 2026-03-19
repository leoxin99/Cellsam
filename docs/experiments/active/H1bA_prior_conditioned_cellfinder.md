# H1bA: Prior-Conditioned CellFinder

> 状态: `implementation complete / formal experiment pending`  
> 负责人: `A1-H1b`  
> 协作: `A1` 主窗口统筹, `A2` 审核, `R1` 复核  
> 关联文档: `docs/conversation_handover/A1/handover_004_2026-03-12_H1b.md`

## 1. 目标

`H1bA` 的目标不是再训练一个完全新的 detector，而是把完整 `DAPI + Actn2` 检测先验接入当前 `CellFinder`，让：

1. `DAPI + Actn2` 负责 cardiomyocyte candidate identity
2. `CellFinder` 负责 whole-cell box extent refinement

当前项目诊断已经很明确：

1. `T27a` 及其 Oracle 分支说明 segmentation branch 已经足够强
2. `CellFinder` 当前真正的瓶颈在 prompt quality，而不是下游 mask quality
3. 因此下一个 detector 方法线应优先做：
   - biology-prior-guided query initialization
   - 而不是继续堆 head-only detector finetune

## 2. Canonical 定义

`H1bA` 当前的正式定义已经收口为：

1. **center-only semantics**
   - 上游只提供 `candidate_points`
   - 以及 `candidate_valid_mask`
   - 不把上游粗框几何当作 runtime 主语义

2. **默认 prior mode**
   - `strict prior-only`

3. **默认 candidate source**
   - `adaptive`

4. **默认 score policy**
   - 对 `prior-aware strict path` 使用 `fixed(0.3)`
   - legacy / no-prior path 保持 `dynamic`

这意味着：

1. `H1bA` 仍然是 `CellFinder` 路线
2. 但它改变的是 query initialization，而不是仅靠阈值微调

## 3. 当前实现范围

### 3.1 已落地代码

1. `src/detection/h1b_priors.py`
   - `adaptive / dapi_cm` 两种 candidate mode
   - 复用完整 `DAPI + Actn2` 先验链

2. `tools/export_h1b_candidate_artifact.py`
   - 导出 prior candidates
   - 已明确 metadata:
     - `box_match_fields_purpose = artifact_audit_only_not_runtime_input`

3. `cellSAM_source/cellSAM/sam_inference.py`
   - `candidate_points_per_image`
   - `candidate_valid_masks`
   - `prior_mode`
   - `score_filter_mode`

4. `cellSAM_source/cellSAM/AnchorDETR/models/transformer.py`
   - prior-conditioned reference point construction

5. `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py`
   - `candidate_valid_mask` 下的 invalid-query logits 抑制

### 3.2 当前没有做的事

1. 还没有正式 detector training 结果
2. 还没有完整多图 downstream E2E 正式结果表
3. 还没有形成 `H1bA` 自己的 locked experiment result artifact

## 4. 已完成的验证

### 4.1 基础 runtime / smoke

已完成：

1. prior-aware runtime smoke
2. candidate/export smoke
3. center-only 语义审计

当前可以下的结论是：

1. `H1bA` 基础代码链已打通
2. 它已不是纯概念提案
3. 但还不能写成“正式训练已验证成功”

### 4.2 center-only quick validation

当前证据支持：

1. `adaptive` 的 center 指标优于 `dapi_cm`
2. 因此默认 source 保留 `adaptive`
3. `dapi_cm` 仅做并行 ablation

关键结论来自 inbox 已审计结果：

- val center `F1`: `adaptive 0.8395 > dapi_cm 0.8191`
- test center `F1`: `adaptive 0.8286 > dapi_cm 0.7995`

### 4.3 runtime source A/B

在相同 `H1bA` 配置下，只切 `adaptive` vs `dapi_cm`：

- val `box_f1@0.3`
  - `adaptive = 0.6520`
  - `dapi_cm = 0.6447`
- test `box_f1@0.3`
  - `adaptive = 0.6374`
  - `dapi_cm = 0.6178`

当前解释：

1. `adaptive` 候选数更多
2. recall 更高
3. 最终 detector-level `F1` 也略高
4. 所以第一版默认 source 继续保留 `adaptive`

### 4.4 threshold / score policy A/B

在 `adaptive + strict prior` 下：

#### detector-level

- val `F1@0.3`
  - `dynamic = 0.5157`
  - `fixed_0p3 = 0.5618`
- test `F1@0.3`
  - `dynamic = 0.5052`
  - `fixed_0p3 = 0.5490`

#### full E2E

- val `E2E F1`
  - `dynamic = 0.3260`
  - `fixed_0p3 = 0.3320`
- test `E2E F1`
  - `dynamic = 0.3113`
  - `fixed_0p3 = 0.3184`

当前结论：

1. `fixed(0.3)` 在 prior-aware strict path 上优于 `dynamic`
2. 提升幅度一致但不大
3. 可以作为默认值修正
4. 不能把它写成 deployment-level major gain

## 5. 当前最稳的结论

当前 `H1bA` 允许写下的结论只有这些：

1. `H1bA` 是一个已经落地到代码层的 prior-conditioned CellFinder 路线
2. canonical 语义应保持 `center-only`
3. 第一版默认 source 保留 `adaptive`
4. prior-aware strict path 的默认 score policy 可改为 `fixed(0.3)`
5. `H1bA` 目前仍处于：
   - `implementation complete`
   - `formal experiment pending`

当前不允许写成：

1. `H1bA 已经证明优于现有 CellFinder 主线`
2. `H1bA 已经是 locked detector result`
3. `dapi_cm` 已经应替代 `adaptive`

## 6. 待做事项

### P0

1. 形成正式 `H1bA` detector-level 评估结果
   - val71
   - test73
   - 固定 source:
     - `adaptive`
     - `dapi_cm`

2. 形成正式 downstream E2E 结果
   - segmentation parent 先用 `T27a`
   - 对比：
     - raw `CellFinder`
     - `H1bA adaptive`
     - `H1bA dapi_cm`

3. 将结果收敛成一个 locked result artifact

### P1

4. 若 `strict prior-only` 表现不足，再测试：
   - `hybrid` fallback

5. 若 detector-level 与 E2E 都有正信号，再考虑 prior-aware training

## 7. 推荐执行顺序

1. 先完成 `H1bA` 正式 detector-level 结果
2. 再接 `T27a` 跑 downstream E2E
3. 如果仍然只见 detector-level gain、不见 E2E gain：
   - 记录为重要负结果
   - 不直接升主线
4. 如果 E2E 也改善：
   - 再决定是否做正式 training branch

## 8. 与其他主线的关系

### 和 `S2-E3`

`S2-E3` 的问题是：

1. 继续沿用现有 `CellFinder` artifact
2. 只改 segmentation training

如果 `S2-E3` 继续弱，最自然的下一步不是继续堆 recipe，而是转向 `H1bA`：

- 直接改善 detector-produced prompts

### 和 `H1a`

`H1a` 当前保留为：
- probe / control branch

`H1bA` 则是：
- 更有希望的 detector 主方法线

## 9. 当前一句话总结

> `H1bA` 已经从“概念方案”进入“可正式实验”的状态；当前默认配置应保持 `center-only + adaptive + strict prior + fixed(0.3)`，下一步是形成正式 detector-level 与 downstream E2E 结果，而不是继续停留在 smoke / inbox 结论层。
