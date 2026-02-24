# Task Backlog (Active)

> 状态: 🟢 Active  
> 维护原则: 只记录"可执行任务"，每项必须有口径/产物/完成标准。  
> 更新时间: 2026-02-20 (2.19 导师会议后重排优先级)

## 📋 目录 (按优先级排列)

### 🔴 P0 — 导师会议紧急任务
- [T16. Baseline 对比实验](#t16-baseline-对比实验--p0)
- [T17. Training Curves 图](#t17-training-curves-图--p0)
- [T18. 三通道 Decoder 实验](#t18-三通道-decoder-实验-p0)

### 🟡 P1 — 论文消融 / 调研
- [T19. 框外像素分割策略调研](#t19-框外像素分割策略调研-p1)
- [T21. 研究 CellSAM 原始 loss 设定](#t21-研究-cellsam-原始-loss-设定-)
- [T12. Phase 1 Loss 消融实验](#t12-phase-1-loss-消融实验-论文-ablation-table-)

### 🟢 P2 — 有时间就做
- [T20. Grad-CAM 可解释性可视化](#t20-grad-cam-可解释性可视化-p2)
- [T11. Encoder LoRA 微调探索](#t11-encoder-lora-微调探索)
- [T13. Adapter vs BF 公平对比](#t13-adapter-vs-bf-公平对比-原-t7)
- [T9. dataset_parameters.md 更新](#t9-dataset_parametersmd-剩余章节更新)
- [T10. CLAUDE.md 关联文档审核](#t10-claudemd-关联文档逐个深度审核)

### ⛔ P3 / 暂缓
- [T5. P2-A N/O Loss (已终止)](#t5-p2-a-no-loss-已终止)
- [T14/T15. P2-B 冲突诊断](#t1415-p2-b-冲突诊断实验)
- [T7. 三通道 Adapter 对比](#t7-三通道-adapter-对比实验)
- [T8. 推理冲突区域策略](#t8-推理冲突区域高级策略探索)

### ✅ 已完成
- [T1. E34 Test 封板](#t1-e34-test-封板评估-dapiadaptive--completed-2026-02-14) | [T2. E34b 联合消融](#t2-e34b-边缘双核联合消融-val71--completed-2026-02-14) | [T3. Adaptive 退化诊断](#t3-adaptive-退化诊断补充--completed-2026-02-16)
- [T4. 参数防呆](#t4-默认参数与锁定参数的执行防呆--completed-2026-02-16) | [T3b. 半径重扫](#t3b-adaptive-search_radius-重扫-80-180--completed-2026-02-19)

---

## 0. 🔴 导师会议紧急任务 (2026-02-19 会议决策)

> **来源**: `docs/meeting_notes_2.19.md`
> **原则**: 下列任务直接影响论文可信度，应优先于所有内部优化任务执行。

### T16. Baseline 对比实验 🔴 P0
- 优先级: **P0** (导师: "非常重要，一定要对比")
- 目标: 与 Cellpose / StarDist / MedSAM / SAMCell 在 test(73) 上统一评估
- 方案: 此前 A2 已设计 E-B1~B6 实验清单 (见 `agent_inbox.md [2026-02-18 21:46]`)
  - **Group A** (Oracle GT boxes): CellSAM 原始 (E-B4) / MedSAM (E-B5)
  - **Group B** (各自检测 E2E): Cellpose (E-B1) / StarDist (E-B2) / SAMCell (E-B6)
- 产物:
  - `tools/baseline_eval.py`
  - `experiments/baseline_comparison/results.json`
  - 论文对比表 (Group A + Group B)
- 完成标准:
  - [ ] E-B4 CellSAM 原始 test(73) 评估
  - [ ] E-B5 MedSAM test(73) 评估
  - [ ] E-B1 Cellpose test(73) 评估
  - [ ] E-B2 StarDist test(73) 评估
  - [ ] E-B6 SAMCell test(73) 评估
  - [ ] 汇总为论文对比表

### T17. Training Curves 图 🔴 P0
- 优先级: **P0** (导师: "这个我是要看的，比较标准")
- 目标: 提供 Phase 1 训练过程的 epochs vs loss/PQ 曲线 (train + val)
- 方案:
  - 从已有训练日志中提取数据 (e.g., `checkpoints/E_phase1_rebalance_l4/` log)
  - 绘制标准曲线: x=epoch, y=loss / PQ / Dice (train + val 两条线)
- 产物:
  - `figures/training_curves_phase1.png`
  - 论文中作为训练过程证据
- 完成标准:
  - [ ] 从训练日志提取 epoch-by-epoch 指标
  - [ ] 生成 loss curve + metric curve 图
  - [ ] 确认 train vs val gap 合理 (非严重 overfitting)

### T18. 三通道 Decoder 实验 P0
- 优先级: **P0** (导师: "必须做，否则不太有信服力")
- 目标: 用 Phase 1 最优 loss 配置，只改通道输入，验证多通道效果
- 方案:
  - 实验 1: BF + Actn2 (2ch) — `use_bf_only: false`, 映射 BF→R, Actn2→G, 空→B
  - 实验 2: BF + DAPI + Actn2 (3ch) — 三通道全用
  - 固定: Phase 1 所有 loss/lr/epochs 不变，只改输入
  - 只做 Decoder fine-tuning
- 产物:
  - `src/config/phase1_2ch.yaml` / `phase1_3ch.yaml`
  - `checkpoints/E_2ch_*/best_model.pt` / `E_3ch_*/best_model.pt`
  - Oracle(test) + E2E(test) 评估
- 完成标准:
  - [ ] 2ch 训练 + test(73) 评估
  - [ ] 3ch 训练 + test(73) 评估
  - [ ] 与 Phase 1 BF (PQ=0.464) 对比
  - [ ] 结论: 多通道是否改善边界/是否改善圆化

### T19. 框外像素分割策略调研 P1
- 优先级: **P1** (导师: "这个是会是一个很好的亮点")
- 目标: 了解 CellSAM 推理时框外像素的处理机制，探索改进策略
- 方案:
  - 阅读 CellSAM 推理代码 (`inference/core.py`) 中框外预测的权重逻辑
  - 理解冲突裁决机制 (argmax_prob)
  - 小实验: 调整框外权重/裁决阈值，看边界质量变化
- 产物:
  - 机制描述文档 (可写进论文)
  - 1 个调整实验结果
- 完成标准:
  - [ ] 理清 SAM 框外预测机制
  - [ ] 完成至少 1 个策略调整实验
  - [ ] 决定是否作为论文亮点

### T20. Grad-CAM 可解释性可视化 P2
- 优先级: **P2** (导师: "看你有没有时间")
- 目标: 用 Grad-CAM 可视化模型关注区域，作为多通道实验的解释证据
- 方案: 对 encoder 最后一层做 Grad-CAM，project 到输入图像
- 完成标准:
  - [ ] Grad-CAM 实现
  - [ ] BF vs 3ch 的注意力区域对比图

---

## 1. Short-Term (已完成)

### T1. E34 Test 封板评估 (DAPI/Adaptive) ✅ Completed (2026-02-14)
- 优先级: P0
- 结果:
  - DAPI: P=0.7462, R=0.8699, F1=0.8033
  - Adaptive: P=0.6968, R=0.8123, F1=0.7502
  - winner: DAPI (`experiments/ablation_detection_lock/results.json`)

### T2. E34b 边缘/双核联合消融 (val71) ✅ Completed (2026-02-14)
- 优先级: P0
- 结果:
  - 最优: `edge_margin=20`, `size_ratio_threshold=2.5`, `merge_coeff=1.4`
  - 指标: P=0.7639, R=0.8633, F1=0.8106

### T3. Adaptive 退化诊断补充 ✅ Completed (2026-02-16)
- 优先级: P1
- 结果:
  - 诊断结论: `cause_code=zline_saturated`

### T4. 默认参数与锁定参数的执行防呆 ✅ Completed (2026-02-16)
- 优先级: P1
- 结果: profile 机制已上线

### T3b. Adaptive `search_radius` 重扫 (80-180) ✅ Completed (2026-02-19)
- 优先级: P1
- 结果:
  - 最优 `search_radius=160`, F1=`0.7800`

---

## 2. Short-Term (论文消融 — 优先级降为会议任务之后)

### T21. 研究 CellSAM 原始 loss 设定 🆕
- **执行者**: A1
- **优先级**: **P1** (论文 loss motivation 需要)
- **目标**: 搞清 CellSAM 原始论文/代码使用什么 loss，作为我们改动的对比基线
- **调研内容**:
  - CellSAM 原始训练 loss 类型 (BCE? Dice? Focal? IoU?)
  - loss 权重/超参数设置
  - 是否有 boundary/AJI 等辅助 loss
  - 论文中 loss 的 motivation (为什么这样选)
- **来源**: CellSAM 原始论文 + GitHub 代码库
- **产物**: 简短总结 (可写入 `codex_claude_seg.md` 或单独文档)
- **完成标准**:
  - [ ] 找到原始 loss 定义 (代码 + 论文)
  - [ ] 与我们当前 CombinedLoss 对比差异
  - [ ] 为论文 §Loss Design 提供 motivation 依据

### T12. Phase 1 Loss 消融实验 (论文 Ablation Table) 🆕
- **执行者**: A2
- **优先级**: **P1** (论文需要，但在 Baseline/3ch 之后)
- **背景**: Phase 1 同时改了 4 个变量，需逐变量消融表。
- **目标**: 建立完整的 loss 消融表

#### 基准配置 (Baseline = Phase 1)
```yaml
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
loss:
  pos_weight: 2.0
  boundary_weight: 1.5
  use_boundary: true
  use_aji: true
  aji_weight: 0.2
  use_contour: true
  contour_weight: 0.3
```

#### 第一层: 开/关消融 (必做, 5 个实验)

| ID | 实验 | 改动 | 目的 |
|----|------|------|------|
| Ab-1 | 关 BoundaryLoss | `use_boundary: false` | Boundary 的贡献 |
| Ab-2 | 关 ContourLoss | `use_contour: false` | Contour 的贡献 |
| Ab-3 | 关 AJI Loss | `use_aji: false` | AJI 的贡献 |
| Ab-4 | 关 PQ 早停 | `use_pq_early_stop: false` | PQ 早停的贡献 |
| Ab-5 | 恢复 pos_weight=10 | `pos_weight: 10.0` | pos_weight 降低的贡献 |

#### 第二层 + 第三层 (详见历史版本)
- Ab-6~10 权重级消融
- P2-D (lr=5e-5), P2-E (epochs=80)

#### 完成标准
- [ ] Ab-1~5 (第一层) 训练 + Oracle(test73) 评估完成
- [ ] Ab-6~10 (第二层) 至少完成 Ab-6, Ab-8
- [ ] P2-D, P2-E 训练 + 评估完成
- [ ] 汇总为论文 Ablation Table

### T11. Encoder LoRA 微调探索
- 优先级: **P2** (导师: "时间有就做，没有放 future work")
- 降级原因: 导师明确排序 — 三通道 > 框外策略 > LoRA
- 目标: SAM ViT-B Q/V attention LoRA (rank=4~8)，做一次最简验证
- 完成标准:
  - [ ] LoRA 实现 + 训练完成
  - [ ] PQ 是否超过 Phase 1 (0.475)

### T13. Adapter vs BF 公平对比 (原 T7)
- 优先级: P2 (可与三通道实验合并)
- 背景: E30/E32 旧 checkpoint 无意义，需用 Phase 1 配置重训
- 完成标准:
  - [ ] `adapter_phase1_fair.yaml` → 训练 → test(73) 评估

---

## 2.5. Mid-Term (Phase 2 — 暂缓)

### T5. P2-A N/O Loss (已终止)
- 状态: ⛔ 终止 (Fix1-3 均证实 N/O loss 退化)
- 论文定位: "Preliminary Exploration: 负结果 + 诊断"
- **导师决定**: 不需要深入量化，写"尝试过且退化"即可

### T14/T15. P2-B 冲突诊断实验
- 优先级: **P3** (推迟 / future work)
- 导师: "如果 run 了一下结果不好，那这样就可以了"

---

## 3. Long-Term (Phase 3+ / Future Work)

### T7. 三通道 Adapter 对比实验
- 优先级: P2 → 可能被 T18 覆盖
- 目标: BF-only vs Adapter (三通道映射) 在统一推理口径下对比

### T8. 推理冲突区域高级策略探索
- 优先级: P3

---

## 3.5 Documentation Audit

### T9. `dataset_parameters.md` 剩余章节更新
- 优先级: P1 → 降为 **P2** (论文实验优先)

### T10. CLAUDE.md 关联文档逐个深度审核
- 优先级: P1 → 降为 **P2** (论文实验优先)
- 待审核清单:

  | # | 文档 | 状态 |
  |---|------|------|
  | 10a | `docs/inference_standard.md` | [ ] |
  | 10b | `docs/dapi_detection_design.md` | [ ] |
  | 10c | `docs/code_inventory.md` | [ ] |
  | 10d | `docs/experiments_log.md` | [ ] |
  | 10e | `docs/naming_convention.md` | [ ] |
  | 10f | `docs/error_log_and_checklist.md` | [ ] |
  | 10g | `docs/alice_quick_reference.md` | [ ] |
  | 10h | `docs/phase2_design.md` | [ ] |
  | 10i | `docs/progress_timeline_2.13.md` | [ ] |
  | 10j | `CLAUDE.md` | [ ] |

---

## 4. Completed Recently

- ✅ E34b val(71) 联合消融完成 (F1=0.8106)
- ✅ E34 test(73) 封板完成 (DAPI F1=0.8033)
- ✅ T3 Adaptive 退化诊断 (zline_saturated)
- ✅ T4 Profile 防呆机制上线
- ✅ T3b 半径重扫 (search_radius=160, F1=0.7800)
- ✅ detection profiles 统一为 locked_eval (runtime_default 已移除)
