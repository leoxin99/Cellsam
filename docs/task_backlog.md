# Task Backlog (Active)

> 状态: 🟢 Active  
> 维护原则: 只记录“可执行任务”，每项必须有口径/产物/完成标准。  
> 更新时间: 2026-02-19

---

## 1. Short-Term (本周优先)

### T1. E34 Test 封板评估 (DAPI/Adaptive) ✅ Completed (2026-02-14)
- 优先级: P0
- 目标: 固定 val(71) 锁定候选参数，在 test(73) 上仅执行一次最终评估，不反向调参。
- 输入:
  - DAPI 候选: `min=1500, max=20000, relative_1.2x`
  - Adaptive 候选: `radius=200, min_zlines=5, zline_threshold=0.01`
- 产物:
  - `experiments/ablation_detection_lock/results.json`
  - 文档回填: `docs/dapi_detection_design.md`, `docs/experiments_log.md`, `CLAUDE.md`
- 完成标准:
  - [x] test(73) 结果写盘
  - [x] 文档标记 “参数已封板”
- 结果:
  - DAPI: P=0.7462, R=0.8699, F1=0.8033
  - Adaptive: P=0.6968, R=0.8123, F1=0.7502
  - winner: DAPI (`experiments/ablation_detection_lock/results.json`)

### T2. E34b 边缘/双核联合消融 (val71) ✅ Completed (2026-02-14)
- 优先级: P0
- 目标: 在 val(71) 上联合重调 `edge_margin`, `size_ratio_threshold`, `merge_coeff`。
- 背景: 当前三项仍沿用历史经验值，尚未纳入统一口径 val 锁定。
- 建议搜索空间:
  - `edge_margin`: [20, 32, 50]
  - `size_ratio_threshold`: [2.0, 2.5, 3.0, 3.5]
  - `merge_coeff`: [1.0, 1.2, 1.4, 1.5]
- 产物:
  - `experiments/ablation_detection_e34b/results.json`
  - [x] 已固定 E34b 参数用于 test 封板
- 结果:
  - 最优: `edge_margin=20`, `size_ratio_threshold=2.5`, `merge_coeff=1.4`
  - 指标: P=0.7639, R=0.8633, F1=0.8106
- 完成标准:
  - 输出 micro P/R/F1（IoU=0.3）
  - 记录最优参数和二优参数，避免偶然点

### T3. Adaptive 退化诊断补充 ✅ Completed (2026-02-16)
- 优先级: P1
- 目标: 用 `adaptive_ratio/fallback_count/mean_zlines` 判断 B2/B3 不敏感原因。
- 产物:
  - `experiments/ablation_adaptive_val/results.json` 中诊断字段
  - 诊断摘要写入 `docs/dapi_detection_design.md`
- 完成标准:
  - [x] 明确是“参数确实不敏感”还是“大量 fallback 导致”
- 结果:
  - B2 (`min_zlines`) F1 区间: `0.7472 -> 0.7472` (range=`0.0000`)
  - B3 (`zline_threshold`) F1 区间: `0.7459 -> 0.7472` (range=`0.0013`)
  - `adaptive_ratio=1.0`, `fallback_count=0`（全程无 fallback）
  - 诊断结论: `cause_code=zline_saturated`（当前半径下 Z-line 信号饱和，不是 fallback 退化）
- 结果文件:
  - `experiments/ablation_adaptive_val/results.json` (`diagnosis_t3`)
  - `experiments/ablation_adaptive_val/diagnosis_t3.json`

### T4. 默认参数与锁定参数的执行防呆 ✅ Completed (2026-02-16)
- 优先级: P1
- 目标: 降低“误用默认参数”风险。
- 方案:
  - 推理/评估脚本显式打印当前 detection 参数
  - 引入 `profile` 机制（`runtime_default` / `locked_eval`）
- 完成标准:
  - [x] 关键脚本输出参数快照
  - [x] 文档写清 profile 选择规则
- 结果:
  - 新增统一 profile 模块: `src/detection/profiles.py`
  - `evaluate_e2e.py` 接入 `--detection-profile`（默认 `locked_eval`）
  - `ablation_detection_lock.py` / `ablation_detection_e34b.py` / `ablation_adaptive_val.py` 接入 `--profile`
  - `docs/inference_standard.md` 新增 profile 执行规则（4.1）

### T3b. Adaptive `search_radius` 重扫 (80-180) ✅ Completed (2026-02-19)
- 优先级: P1
- 背景:
  - T3 已确认 `zline_saturated`（非 fallback）；
  - R1 在 2026-02-18 修正优先级：该任务不应降级到 backlog。
- 目标:
  - 在 `val(71)` 下验证缩小半径后是否恢复 B2/B3 敏感性并提升 Adaptive 检测质量。
- 协议:
  - 固定 `locked_eval` 口径：`min/max_nucleus_area=1500/20000`
  - B1 半径候选: `[80, 100, 120, 140, 160, 180]`
  - B2/B3 仅在新最优半径上执行（避免全量无效搜索）
- 产物:
  - `experiments/ablation_adaptive_radius_val/results.json`（新目录，避免覆盖已封板历史）
  - 文档回填: `docs/dapi_detection_design.md`, `docs/experiments_log.md`
- 完成标准:
  - 明确 `mean_zlines` 是否脱离饱和区间
  - 明确 `adaptive_ratio/fallback_count` 是否进入可调区间
  - 结论写明: 保留 Adaptive 路线 / 仅作为对照路线
- 结果:
  - B1 最优半径: `search_radius=160`, F1=`0.7788`（`80-180` 区间内最佳）
  - B2 (`min_zlines`) 仍近似不敏感: F1 区间 `0.7788 -> 0.7788`（range=`0.0000`）
  - B3 (`zline_threshold`) 出现轻微敏感: 最优 `0.05`, F1=`0.7800`
  - 诊断: `adaptive_ratio` 持续接近 `1.0`、`fallback_count` 接近 `0`，仍以自适应分支为主
- 结果文件:
  - `experiments/ablation_adaptive_radius_val/results.json`
  - `tools/ablation_adaptive_val.py` 已补 `--b1-values` + `--output-dir`，并修复 profile 参数未实际传入的 bug

---

## 2. Short-Term (本周优先 — 新增)

### T11. Encoder LoRA 微调探索 🆕
- 优先级: **P1** (短期)
- 背景: 当前只微调 Decoder (~4M 参数)，Encoder (86M) 完全冻结。SAC 论文证明 LoRA 在 Encoder attention 层有效。如果 PQ 卡在 0.50 以下，LoRA 是最自然的下一步
- 目标: 在 SAM ViT-B 的 Q/V attention 层插入 LoRA (rank=4~8)，验证是否提升 PQ/Dice
- 方案:
  - 创建 `src/config/phase2f_lora.yaml` (基于 Phase 1 config + LoRA 开关)
  - 修改 `src/train.py` 支持 LoRA 注入 (使用 `peft` 库或手动插入低秩矩阵)
  - 训练参数: lr=5e-5 (LoRA 部分用更小 lr), epochs=50
- 产物:
  - `checkpoints/E_phase2f_lora/best_model.pt`
  - Oracle(test) + E2E(test) 评估结果
- 完成标准:
  - [ ] LoRA 实现 + 训练完成
  - [ ] PQ 是否超过 Phase 1 (0.475)
  - [ ] 决定是否纳入论文消融表

### T12. Phase 1 Loss 消融实验 (论文 Ablation Table) 🆕🔴 HIGH PRIORITY
- **执行者**: A2
- **优先级**: **P0** (论文 Table 必需)
- **背景**: Phase 1 同时改了 4 个变量 (pos_weight/boundary_weight/contour_weight/pq_early_stop)，无法归因各自贡献。论文需要逐变量消融表。
- **目标**: 建立完整的 loss 消融表，证明每个组件的贡献

#### 基准配置 (Baseline = Phase 1)
```yaml
# 文件: src/config/phase1_rebalance_l4.yaml — 所有消融实验的基准
# 以下参数在所有消融中保持不变 (除被消融的变量外):
data:
  target_size: [1024, 1024]
  use_bf_only: true
model:
  freeze_encoder: true
  use_adapter: false
training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.0001
  weight_decay: 0.0001
  warmup_epochs: 5
  early_stop_patience: 15
  use_pq_early_stop: true
optimizer:
  type: "adamw"
  scheduler: "cosine_warmup"
# === 以下为被消融的 loss 参数 (Phase 1 最优值) ===
loss:
  pos_weight: 2.0
  boundary_weight: 1.5
  use_boundary: true
  use_aji: true
  aji_weight: 0.2
  use_contour: true
  contour_weight: 0.3
```

#### 控制变量规则
> **每个消融实验只改一个变量**，其余完全复用 Phase 1 基准。
> 评估指标: Oracle(test, 73) BM-1to1 Dice + PQ + AJI。

#### 第一层: 开/关消融 (必做, 5 个实验)

| ID | 实验 | 改动 | 配置文件 | 目的 |
|----|------|------|---------|------|
| Ab-1 | 关 BoundaryLoss | `use_boundary: false` | `ablation_no_boundary.yaml` | Boundary 的贡献 |
| Ab-2 | 关 ContourLoss | `use_contour: false` | `ablation_no_contour.yaml` | Contour 的贡献 |
| Ab-3 | 关 AJI Loss | `use_aji: false` | `ablation_no_aji.yaml` | AJI 的贡献 |
| Ab-4 | 关 PQ 早停 | `use_pq_early_stop: false` | `ablation_no_pqstop.yaml` | PQ 早停的贡献 |
| Ab-5 | 恢复 pos_weight=10 | `pos_weight: 10.0` | `ablation_posw10.yaml` | pos_weight 降低的贡献 |

#### 第二层: 权重级消融 (建议做, 5 个实验)

| ID | 实验 | 改动 | 目的 |
|----|------|------|------|
| Ab-6 | boundary_weight=0.5 (E29 值) | 恢复旧值 | 确认 1.5 vs 0.5 |
| Ab-7 | boundary_weight=3.0 | 进一步提高 | 是否越高越好 |
| Ab-8 | contour_weight=0.1 (原值) | 恢复旧值 | 0.3 vs 0.1 |
| Ab-9 | aji_weight=0.5 | 提高 | AJI 是否被低估 |
| Ab-10 | pos_weight=5.0 | 中间值 | 10→5→2 最优点 |

#### 第三层: 超参数消融 (已有 P2-D/E)

| ID | 实验 | 改动 |
|----|------|------|
| P2-D | lr=5e-5 | 其余不变 |
| P2-E | epochs=80 | 其余不变 |

#### ⚠️ 已知待审计参数
以下参数为**硬编码默认值**, 无文档化理由, 需在消融中标注:
- `BoundaryLoss(boundary_width=3)` — 3px 侵蚀核, 来源未记录
- `raw_base = 0.3` — base loss 权重地板, 来源: `max(0.3, 1 - total_extra)`
- `AJILoss(smooth=1.0)` — Jaccard 平滑项

#### 完成标准
- [ ] Ab-1~5 (第一层) 训练 + Oracle(test73) 评估完成
- [ ] Ab-6~10 (第二层) 至少完成 Ab-6, Ab-8
- [ ] P2-D, P2-E 训练 + 评估完成
- [ ] 汇总为论文 Ablation Table (每行一个实验, 列: PQ / BM-Dice / AJI)
- [ ] 结果写入 `experiments_log.md`

#### GPU 时间估算
- 第一层: 5 × 4.5h = **22.5h**
- 第二层: 5 × 4.5h = **22.5h** (按需)
- 第三层: 2 × 4.5h = **9h**

### T13. Adapter vs BF 公平对比 (原 T7, 已重定义) 🆕
- 优先级: P1
- 背景: E30/E32 旧 checkpoint 仅用 10% 数据 + 旧 loss 配置 (pos_weight=10, boundary=0.5, 无 PQ 早停)，与 Phase 1 配置差距极大，直接评估无意义
- **正确方案**: 用 Phase 1 最优配置 (`phase1_rebalance_l4.yaml`)，仅改 `use_adapter=true` + `use_bf_only=false`，重新训练 Adapter 版本
- 完成标准:
  - [ ] 创建 `adapter_phase1_fair.yaml` (基于 phase1_rebalance_l4，改 adapter=true, bf_only=false)
  - [ ] 全量训练 + test(73) 评估 (Oracle PQ/BM-Dice)
  - [ ] 与 Phase 1 BF (PQ=0.4641) 公平对比

### T14. P2-B 诊断实验 A: P1 冲突量化 🆕
- 优先级: P1
- 目标: 量化 Phase 1 模型的 conflict_rate 和 intrusion_rate
- 方案: 在 test(73) 跑 `comprehensive_eval.py`，已有 `conflict_pixels` 字段，补充 `intrusion_rate` 计算
- 完成标准:
  - [ ] conflict_rate = conflict_pixels / total_fg_pixels
  - [ ] intrusion_rate = 被错误分配的像素 / 总 instance 像素
  - [ ] 明确冲突的实际影响大小

### T15. P2-B 诊断实验 B: Oracle 冲突消解 🆕
- 优先级: P1
- 目标: 如果用 GT mask 强制消解冲突区域，PQ 变化多少？
- 方案: 在冲突像素上用 GT 归属替换 argmax_prob → 重算 PQ
- 完成标准:
  - [ ] PQ(oracle conflict) vs PQ(argmax_prob) delta
  - [ ] 如果 delta ≈ 0 → 冲突不是瓶颈，P2-B 无必要
  - [ ] 如果 delta > 0.02 → 冲突有改善空间，P2-B 全局版值得尝试

## 2.5. Mid-Term (Phase 2)

### T5. P2-A 训练与评估闭环
- 优先级: P1
- 目标: 完成 P2-A 训练、Oracle/E2E 评估、回归验证。
- 修复路径: 参照 [`phase2_design.md §7.5`](file:///d:/AI/paper/CellSam/docs/phase2_design.md) 的 fix1-fix4
- 完成标准:
  - fix1 (从 P1 微调) 已执行，config 中 `checkpoint` 指向 P1 best
  - 训练完成且无关键报错
  - 指标与 Phase 1 对比完成（PQ 不低于 P1 baseline 0.475）
  - 若 PQ 仍退化，需追查 loss 设计（fix2/fix3/fix4），不可标记为"完成"

### T6. P2-B 决策门
- 优先级: P2
- 目标: 基于 P2-A 结果决定是否进入 P2-B（Contour/Topology/学习率路线）。
- 完成标准:
  - 有明确 go/no-go 结论与理由

---

## 3. Long-Term (Phase 3+)

### T7. 三通道 Adapter 对比实验
- 优先级: P2
- 目标: BF-only vs Adapter (三通道映射) 在统一推理口径下对比。
- 完成标准:
  - 统一脚本与统一指标
  - 结果可直接用于论文表格

### T8. 推理冲突区域高级策略探索
- 优先级: P3
- 目标: 评估 soft-boundary / watershed / CRF(MRF) 对冲突像素归属的影响。
- 完成标准:
  - 至少一个策略实现可复现实验

---

## 3.5 Documentation Audit (逐文档深度审核)

> 背景: 之前对文档的批量更新不够细致，存在过时参数、口径不一致等遗留问题。
> 原则: 每个文档独立审核，逐节对照源码/config/实验结果，确保无陈旧/错误信息。

### T9. `dataset_parameters.md` 剩余章节更新
- 优先级: P1（随 E34b 结果一起做）
- 前置: T2 (E34b) 完成后才有 val 复核数据
- 待更新章节:
  - §6 边缘过滤参数: 补充 val(71) 复核小节 (`edge_margin` 20/32/50)
  - §7 双核合并参数: 补充 val(71) 复核小节 (`merge_coeff + size_ratio_threshold` 联合)
  - §9 框扩展参数: 区分 DAPI-only vs Adaptive fallback 扩展逻辑，对照 `dapi.py` 函数签名
  - §11 后处理参数: 标注 SSOT 为 `inference_standard.md`，本节仅保统计依据
- 完成标准:
  - 每节参数值与 `dapi.py` / config YAML 一致
  - §12 更新方案表全部标 ✅ Done

### T10. CLAUDE.md 关联文档逐个深度审核
- 优先级: P1
- 目标: 对 CLAUDE.md 核心文档表中的每个 🟢 Active 文档做独立审核，消除过时/不一致信息。
- 审核方式: 每篇独立过一遍，逐节对照源码、config、实验结果。
- 待审核清单:

  | # | 文档 | 审核重点 | 状态 |
  |---|------|---------|------|
  | 10a | `docs/inference_standard.md` | 推理参数与 `src/inference/core.py` 一致性 | [ ] |
  | 10b | `docs/dapi_detection_design.md` | 检测参数锁定表与 E34 结果、代码默认值一致 | [ ] |
  | 10c | `docs/code_inventory.md` | 文件路径/入口是否仍存在、描述是否准确 | [ ] |
  | 10d | `docs/experiments_log.md` | 实验编号连续性、结果数值与日志一致 | [ ] |
  | 10e | `docs/naming_convention.md` | 命名规则与实际代码/文件命名一致 | [ ] |
  | 10f | `docs/error_log_and_checklist.md` | 已解决问题标记完成、新问题补录 | [ ] |
  | 10g | `docs/alice_quick_reference.md` | SLURM 模板/module 名与当前集群环境一致 | [ ] |
  | 10h | `docs/phase2_design.md` | §7 P2A 分析已修正，其余章节是否有陈旧信息 | [ ] |
  | 10i | `docs/progress_timeline_2.13.md` | 时间线节点与实际完成时间一致 | [ ] |
  | 10j | `CLAUDE.md` 自身 | 核心文档表、参数分工表、阶段摘要与最新状态一致 | [ ] |

- 完成标准:
  - 每篇审核后在上表标 [x]
  - 发现的问题就地修复（不积压）
  - 每篇更新后在该文档的更新日志中追加条目

## 4. Completed Recently

- ✅ E34b val(71) 联合消融完成:
  - 最优: `edge_margin=20`, `size_ratio_threshold=2.5`, `merge_coeff=1.4`, F1=0.8106
- ✅ E34 test(73) 单次封板完成:
  - DAPI: F1=0.8033
  - Adaptive: F1=0.7502
  - winner: DAPI
- ✅ T3 Adaptive 退化诊断完成:
  - 结论: B2/B3 不敏感的主因是 `zline_saturated`，非 fallback 导致
- ✅ T4 默认参数与锁定参数防呆完成:
  - profile 机制已上线，关键检测评估脚本默认 `locked_eval`
- ✅ T3b 半径重扫完成:
  - 最优 `search_radius=160`, `min_zlines=5`, `zline_threshold=0.05`, F1=`0.7800`
- ✅ 检测消融脚本 GT 面积过滤移除（不再静默修改 GT 分母）
- ✅ `ablation_adaptive_val.py` 支持 `--stage` + `--resume` + detector 参数一致性校验
