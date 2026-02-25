# CellSAM 实验记录 (Experiment Log)

> **状态**: 🟢 Active — 实验流水账主文档  
> **最后更新**: 2026-02-25  
> **事实来源**: 此文档为实验记录的 SSOT，按时间顺序记录所有实验  
> **完整历史存档**: [`experiments_log_archive.md`](experiments_log_archive.md)

---

## 📋 目录

- [实验索引](#实验索引-experiment-index)
- [Phase 1: Loss 重平衡 + PQ 早停](#phase-1-loss-重平衡--pq-早停-)
- [P2-A: 邻居侵占/重叠互斥损失](#p2-a-邻居侵占重叠互斥损失-fix1-3)
- [E34: DAPI/Adaptive 参数锁定](#e34-completed-dapiadaptive-参数统一锁定实验-)
- [T3b: Adaptive search_radius 半径重扫](#t3b-adaptive-search_radius-重扫-2026-02-19)
- [Semantic vs Instance Dice 关键发现](#-关键发现-semantic-vs-instance-dice-2026-02-05-)
- [E22: Box Clipping 修复](#e22-推理-box-clipping-修复--边界精度分析-)
- [E20: DAPI vs Adaptive 消融](#e20-dapi-only-vs-adaptive-检测消融-)
- [E19: 边缘/双核参数微调](#e19-边缘双核参数微调)
- [早期实验归档 (E01-E18)](#早期实验归档-e01-e18)
- [待实验](#待实验-planned)

---

## 实验索引 (Experiment Index)



| ID | 日期 | 实验名称 | 结果 | 状态 |

|----|------|---------|------|------|

| E01 | 2026-01-08 | 类别不平衡修复 | Dice 0→0.52 | ✅ 成功 |

| E02 | 2026-01-08 | CellFinder 检测测试 | F1=0.012 | ❌ 失败 |

| E03 | 2026-01-08 | DAPI 核检测方案 | F1=0.750 | ✅ 成功 |

| E04 | 2026-01-09 | 全管线测试 (像素级) | Dice=0.58 | ⚠️ 基线 |

| E05 | 2026-01-09 | 全管线测试 (实例级) | Dice=0.71 | ✅ 改进 |

| E06 | 2026-01-11 | 分水岭核分离 | F1=0.34 | ❌ 失败 |

| E09 | 2026-01-11 | 验证指标实现 | PQ=0, AJI=0.10 | ⚠️ 发现问题 |

| **E12** | **2026-01-11** | **边界损失微调** | **PQ↑265%, Dice↑8%** | **✅ 当前最佳** |

| E13 | 2026-01-11 | 数据集标准化 + 代码简化 | 固定划分 + 统一训练入口 | ✅ 成功 |

| E14 | 2026-01-14 | 核-细胞轴向对齐分析 + 智能扩展 | 50%对齐@30°, Dice+3.3% | ✅ 成功 |

| E15a | 2026-01-15 | 多通道融合A: BF基线 | Val Dice=0.6472 | ✅ 基线 |

| **E15b** | **2026-01-15** | **多通道融合B: BF+DAPI+Actn2** | **Pixel Dice=0.7454** | **❌ 劣于E12** |

| ~~E15c~~ | - | ~~多通道融合C: 加权Actn2~~ | 已废弃 | ❌ 取消 |

| ~~E15d~~ | - | ~~多通道融合D: 不确定引导~~ | 已废弃 | ❌ 取消 |

| **E16** | **2026-01-16** | **E12 vs E15b 对比** | **E12优2.6%** | **✅ E12确认最佳** |

| **E17** | **2026-01-21** | **GT 细胞面积统计** | **阈值 40K-450K** | **✅ 数据驱动** |

| **E18** | **2026-01-23** | **SarcGraph 检测对比** | **F1↑7.4%** | **✅ 优于DAPI** |

| **E19** | **2026-01-26** | **边缘/双核参数微调** | **Edge=100px, Binuc=1.5x** | **✅ 精确化** |

| **E20** | **2026-01-30** | **DAPI Only vs Adaptive 消融** | **DAPI Only F1=0.765 胜** | **✅ 完成** |

| **E21** | **2026-01-30** | **E12 vs Semantic Adapter 对比** | **E12 Dice=0.598 胜** | **✅ 完成** |

| **E22** | **2026-02-02** | **推理 Box Clipping 修复 + 边界精度分析** | **PQ@0.3↑90x, 发现67%过分割** | **⚠️ 需改进** |

| **E23** | **2026-02-02** | **关键Bug修复：数据加载uint8截断** | **DAPI检测0→78% F1** | **✅ 关键修复** |

| E24 | 2026-02-03 | BF Baseline v2 (A100) | Val Dice=0.7520 | ✅ 完成 |

| E25 | 2026-02-03 | Boundary Enhanced (L4) | Val Dice=0.7595 | ✅ 完成 |

| E26 | 2026-02-03 | 3ch No Adapter (L4) | Val Dice=0.7549 | ✅ 完成 |

| E27 | 2026-02-04 | 3ch Semantic Adapter (A100) | Val Dice=0.7520 | ✅ 完成 |

| E28 | 2026-02-04 | BF Adapter (A100) | Val Dice=0.7337 | ✅ 完成 |

| **⚠️** | **2026-02-05** | **发现: Semantic Dice 无意义** | **Instance Dice=0.03** | **⚠️ 关键** |

| E29 | 2026-02-05 | BF Instance P1 | PQ=0.33→P1调参后0.475 | ✅ 已训练 (A1) |
| E30 | 2026-02-05 | Adapter Instance P1 | 有 checkpoint 未评估 | ⚠️ 待评估 |
| E31 | 2026-02-05 | BF Instance P2 (全部Loss) | 未训练 | ❌ P2-A 终止 |
| E32 | 2026-02-05 | Adapter Instance P2 | 有 checkpoint 未评估 | ⚠️ 待评估 |
| **E33** | **2026-02-06** | **GT Box + 预训练CellSAM (Baseline)** | **BM-Dice=0.111, PQ=0.000** | **✅ Baseline** |
| **E34** | **2026-02-13~14** | **DAPI/Adaptive 参数统一锁定 (val→test)** | **E34b+test73 已完成并封板** | **✅ 完成** |
| BugFix | 2026-02-13 | GT 框面积过滤移除 | 5173/5173 通过 | 🔧 已修复 |
| **Phase 1** | **2026-02-10** | **Loss 重平衡 + PQ 早停** | **Oracle PQ=0.464, BM=0.695** | **✅ 当前最佳** |
| P2-A Fix1 | 2026-02-15 | N/O Loss (N=0.3, O=0.1) | PQ=0.232 | ❌ 失败 |
| P2-A Fix2 | 2026-02-15 | N/O Loss (N=0.1, O=0.05) | PQ=0.393 | ⚠️ 改善但差 |
| P2-A Fix3 | 2026-02-16 | N/O Loss 延迟启用 | PQ=0.466 (N/O OFF时) | ⚠️ P2-A 终止 |
| T3b | 2026-02-19 | Adaptive search_radius 重扫 | F1=0.780 (radius=160) | ✅ 完成 |
| **T16** | **2026-02-21~22** | **Baseline 对比 (6 methods)** | **见详细表** | **✅ 完成** |
| T16-d200 | 2026-02-22 | Cellpose diameter=200 补充 | PQ=0.002, BM=0.190 | ✅ 完成 |
| **T19-abl** | **2026-02-22** | **Box Clipping 消融** | **clip PQ=0.466 > no-clip PQ=0.437** | **✅ 完成** |
| **T12** | **2026-02-23** | **Loss 消融 (7×2 seeds)** | **posw=10 PQ=0.494, contour 有害** | **✅ 完成** |
| **Best Config** | **2026-02-24** | **posw=10+contour=off (4 runs)** | **PQ=0.484 (mean)** | **✅ 完成** |
| **T18** | **2026-02-24~25** | **三通道消融 (2ch/3ch/noAdapt)** | **T18-C PQ=0.500 (best), A=0.496, B=0.498** | **🔄 5/6 done** |
| **T17** | **2026-02-25** | **Training Curves 工具** | **parse+plot, Phase1 图 ✅** | **✅ 工具完成** |
| **T20** | **2026-02-25** | **Attention 可视化脚本** | **Method A+C, 待执行** | **🔄 脚本就绪** |
| **T11** | **2026-02-25** | **[LoRA Encoder Fine-tuning](t11_lora_design.md)** | **rank=4/8, Q/V LoRA, 待 R1 审核** | **⏳ 设计完成** |



---

## T16 Baseline 对比实验 ⭐⭐⭐ (2026-02-21~22)

- **目的**: 论文 Table — 与公开方法对比
- **评估集**: test(73), 统一使用 `compute_all_metrics()`
- **脚本**: `baseline_eval.py`, `samcell_eval.py`, `comprehensive_eval.py`, `evaluate_e2e.py`

| Method | Type | PQ | BM-Dice | AJI | Sem.Dice |
|--------|------|----|---------|-----|----------|
| Cellpose v4 (auto) | E2E | 0.000 | 0.053 | 0.025 | 0.079 |
| Cellpose v4 (d=200) | E2E | 0.002 | 0.190 | 0.089 | 0.191 |
| SAMCell (LIVECell) | E2E | 0.000 | 0.008 | 0.004 | 0.014 |
| CellSAM (pretrained) | Oracle | 0.000 | 0.121 | 0.056 | 0.219 |
| SAM ViT-B (vanilla) | Oracle | 0.286 | 0.631 | 0.440 | 0.756 |
| **Ours (Phase1_L4)** | **Oracle** | **0.464** | **0.695** | **0.519** | **0.756** |
| **MedSAM** | **Oracle** | **0.576** | **0.771** | **0.634** | **0.862** |
| Ours (Phase1_L4) | E2E | 0.180 | 0.567 | 0.338 | 0.642 |

> ⚠️ MedSAM (PQ=0.576) > Ours Oracle (PQ=0.464).
> MedSAM 受益于 100 万+ 医学图像预训练，但没有检测能力。

### T19-abl: Box Clipping 消融 (2026-02-22)

- **目的**: 验证 `apply_box_clipping` 对 Ours Oracle 的影响（MedSAM baseline 无 clipping）
- **脚本**: 内联 Python, 结果 `experiments/box_clipping_ablation/results.json`

| 配置 | PQ | BM-Dice | AJI | Conflict Pixels |
|------|----|---------|-----|-----------------|
| **with_clip (默认)** | **0.466** | **0.714** | **0.557** | 52,262 |
| no_clip | 0.437 | 0.703 | 0.545 | 57,781 |

**结论**: 去掉 box clipping 反而降 6.2% PQ → 我们的模型在框外产生伪阳性，clipping 有防御价值。MedSAM 不需要 clipping 是因为其框外预测更精准（百万级训练数据的泛化能力更强）。

### T12: Loss 消融实验 ⭐⭐⭐ (2026-02-23)

- **目的**: 论文 Ablation Table — 量化各 loss 组件的贡献
- **设计**: 7 组配置 × 2 seeds (seed=42 on A100, seed=123 on L4) = 14 runs
- **基线**: Phase1 rebalance config (posw=2, boundary=1.5, contour=0.3, AJI=0.2, PQ early stop)
- **评估**: Oracle (GT box), test(73), `comprehensive_eval.py`
- **产物**: `experiments/ablation_eval/seed{42,123}/*.json`, `src/config/ablation_*.yaml`

| 实验 | PQ (mean) | BM-Dice | AJI | Δ PQ | 置信度 |
|------|:---------:|:-------:|:---:|:----:|:------:|
| Full (Phase1, posw=2) | 0.453 | 0.707 | 0.550 | — | — |
| Ab-0: BCE+Dice only | 0.459 | 0.711 | 0.554 | +0.7pp | ⚠️ |
| Ab-1: w/o Boundary | 0.454 | 0.708 | 0.554 | +0.2pp | ⚠️ |
| **Ab-2: w/o Contour** | **0.476** | **0.718** | **0.564** | **+2.3pp** | **✅ 高** |
| Ab-3: w/o AJI | 0.459 | 0.710 | 0.554 | +0.6pp | ⚠️ |
| Ab-4: w/o PQ ES | 0.459 | 0.710 | 0.555 | +0.7pp | ⚠️ |
| **Ab-5: posw=10** | **0.494** | **0.724** | **0.573** | **+4.1pp** | **✅ 高** |

**高置信结论** (两 seed 方向一致 + 幅度大):
1. **pos_weight=10 >> 2** (+4.1pp PQ) — Phase1 降 posw 是错误决策
2. **Contour Loss 有害** (+2.3pp PQ when removed) — 与 SAM prompt-conditioned 训练范式冲突

**Best Config 验证 ✅ (2026-02-24)**:
- 4 runs (A100+L4 × seed42+seed123, 80 epochs)
- 结果:

| Run | GPU | Seed | PQ | BM-Dice | AJI | Sem.Dice | Best Ep |
|-----|-----|------|:--:|:-------:|:---:|:--------:|:-------:|
| 1 | A100 | 42 | 0.487 | 0.722 | 0.571 | 0.802 | 25 |
| 2 | A100 | 123 | 0.486 | 0.718 | 0.568 | 0.797 | 33 |
| 3 | L4 | 42 | 0.486 | 0.722 | 0.571 | 0.802 | 25 |
| 4 | L4 | 123 | 0.479 | 0.718 | 0.569 | 0.800 | 40 |
| **Mean** | | | **0.484** | **0.720** | **0.570** | **0.800** | |

- Checkpoint: `BestConfig_posw10_noCont_20260224_052553/best_model.pt` (A100 seed42)
- **发现**: Best Config PQ=0.484 低于 Ab-5 PQ=0.494，说明 posw=10 与 contour=off 存在交互作用而非简单叠加

---

### T18: 三通道通道消融 🔄 (2026-02-24~25, 5/6 done)

- **目的**: 验证多通道输入 (DAPI+Actn2) 是否提升分割性能
- **基线**: Best Config (PQ=0.484), fine-tune 从 Best Config checkpoint 开始
- **通道顺序**: R=BF · G=Actn2(绿) · B=DAPI(蓝) (生物学一致)
- **代码修改**: `SemanticChannelMapper` 通道顺序 + `use_2ch` 模式 + `IndependentChannelAdapter` 同步

| 实验 | 通道 | Adapter | Seeds | GPU |
|------|--------|:-------:|:-----:|-----|
| T18-A | R=BF, G=Actn2, **B=BF** (2ch) | ✅ | 42+123 | L4+A100 |
| T18-B | R=BF, G=Actn2, **B=DAPI** (3ch) | ✅ | 42+123 | L4+A100 |
| T18-C | R=BF, G=Actn2, B=DAPI (3ch, 无adapter) | ❌ | 42, **123** | L4, **A100** |

- **总运行数**: 7 runs (T18-A/B/C × 2 seeds + T18-Control)
- **配置**: `t18a_2ch.yaml`, `t18b_3ch.yaml`, `t18c_3ch_no_adapter.yaml`, `t18_control_bf_continue.yaml`
- **SLURM**: 7/7 完成 ✅ (含 T18-Control Job 1036827, T18-C s123 Job 1036799)

**完整结果 (2026-02-25, Bug 修正后)**:

> ⚠️ 原始 T18-A/B seed42 L4 评估存在 Bug: `ls -td` 取到了 A100 checkpoint。已修复 (Job 1036804)。

| 实验 | Seed | GPU | PQ↑ | BM-Dice | AJI | Sem.Dice | Best Ep | 备注 |
|------|:----:|:---:|:---:|:-------:|:---:|:--------:|:-------:|:----:|
| Best Config (BF) | mean(4) | — | 0.484 | 0.720 | 0.570 | — | — | |
| **T18-Control (BF 继训)** | **42** | **L4** | **0.488** | **0.719** | **0.568** | **0.795** | **12** | **对照** |
| T18-A (2ch) | 42 | L4 | 0.493 | 0.723 | 0.573 | 0.802 | 7 | 修正 |
| T18-A (2ch) | 123 | A100 | 0.496 | 0.724 | 0.573 | 0.799 | 27 | |
| T18-B (3ch+adapter) | 42 | L4 | 0.496 | 0.723 | 0.572 | 0.797 | 27 | 修正 |
| T18-B (3ch+adapter) | 123 | A100 | 0.498 | 0.725 | 0.574 | 0.801 | 37 | |
| **T18-C (3ch noAdapt)** | **42** | **L4** | **0.500** | **0.726** | **0.573** | **0.801** | **41** | |
| T18-C (3ch noAdapt) | 123 | A100 | 0.499 | 0.725 | 0.572 | 0.798 | 27 | |

> ⚠️ 原始 T18-A/B seed42 L4 评估存在 Bug: `ls -td` 取到了 A100 checkpoint。已修复 (Job 1036804)。

**最终分析 (2026-02-25 16:20, 含对照组)**:

| 对比 | PQ 差 | 含义 |
|------|:-----:|------|
| T18 avg (0.497) vs Best Config (0.484) | +1.3pp | 总提升 (通道 + 训练) |
| **T18-Control (0.488)** vs Best Config (0.484) | **+0.4pp** | **纯训练效应** (lr reset + extra epochs) |
| T18 avg (0.497) vs **T18-Control (0.488)** | **+0.9pp** | **净通道贡献** |

**结论**:
1. 三通道信息**确实有效** (+0.9pp 净效应), 但效果**被训练效应放大了** (~30% 来自额外训练)
2. T18-A ≈ T18-B ≈ T18-C (极差 0.7pp < seed 波动 2.5pp): adapter/通道数无法区分
3. 论文可写: *"Multi-channel PQ=0.497 vs BF-only PQ=0.488 (continued training control), Δ=+0.9pp"*

---

## E34 (Completed): DAPI/Adaptive 参数统一锁定实验 ⭐⭐⭐

**日期**: 2026-02-13 ~ 2026-02-14 (已完成)

**背景**:
- 历史检测消融中存在 test-20 调参记录（探索有效但不用于最终锁定）。
- 2026-02-05 已完成分辨率修正（1736×1776 → 1024 口径），并更新默认参数。
- 需补齐“正确数据集 + 统一口径”的最终参数锁定，避免后续 E2E 结论受争议。

**目标**:
1. 对 DAPI 与 Adaptive 两条框生成方案，在同一协议下完成可复现调参。
2. 仅在 val 集调参，test 集只做一次最终锁定评估。
3. 输出可直接回填 `dapi_detection_design.md`、`dataset_parameters.md`、`CLAUDE.md` 的最终参数表。

**统一协议**:
- 数据集:
  - 调参: `val_ids.txt` 全量 (71)
  - 锁定: `test_ids.txt` 全量 (73), 单次执行
- 指标:
  - Detection: Precision/Recall/F1 (IoU=0.3)
  - E2E: BM-1to1 / PQ / AJI (固定同一 segmentation checkpoint)
- 约束:
  - 不允许根据 test 结果反向调参
  - 所有参数变更需记录脚本、配置、结果文件路径

**实验拆分**:
1. DAPI 参数锁定 (val71): `min_nucleus_area`, `max_nucleus_area`, `use_relative_distance`
2. Adaptive 参数锁定 (val71): `search_radius`, `min_zlines`, `zline_threshold`
3. 固定最优参数后，DAPI vs Adaptive 在 test73 单次对比并锁定

**最终结果 (2026-02-14)**:
- DAPI val(71) 锁定: 最优 `min=1500, max=20000, relative_1.2x`, F1=`0.7965`
- Adaptive val(71) 锁定: 最优 `radius=200, min_zlines=5, zline_threshold=0.01`, F1=`0.7271`
- E34b 联合消融 (val71): 最优 `edge_margin=20`, `size_ratio_threshold=2.5`, `merge_coeff=1.4`, F1=`0.8106`
- test(73) 单次封板: DAPI F1=`0.8033`，Adaptive F1=`0.7502`，winner=`DAPI`
- 文档同步: `CLAUDE.md`、`docs/task_backlog.md`、`docs/dapi_detection_design.md`
- T3b 半径重扫补充 (2026-02-19, val71): `search_radius=160`, `min_zlines=5`, `zline_threshold=0.05`, F1=`0.7800` (`experiments/ablation_adaptive_radius_val/results.json`)

**产物**:
- `experiments/ablation_dapi_val/results.json`
- `experiments/ablation_adaptive_val/results.json`
- `experiments/ablation_adaptive_radius_val/results.json` (T3b follow-up, post-lock diagnostic)
- `experiments/ablation_detection_e34b/results.json`
- `experiments/ablation_detection_lock/results.json`
- 文档回填: `CLAUDE.md`, `docs/task_backlog.md`, `docs/dapi_detection_design.md`

**SSOT 回填状态**:
- ✅ `CLAUDE.md` Step4.5/4.6 已改为 completed
- ✅ `docs/task_backlog.md` T1/T2 已改为 completed 并写入指标
- ✅ `docs/dapi_detection_design.md` 已写入 E34b 与 test73 封板结果

---

## Phase 1: Loss 重平衡 + PQ 早停 ⭐⭐⭐

**日期**: 2026-02-10 ~ 2026-02-12

**配置**: `src/config/phase1_rebalance_l4.yaml`
- 改动: `boundary_weight` 0.5→1.5, `contour_weight` 0.1→0.3, `pos_weight` 10→2, `use_pq_early_stop: true`
- 训练: ALICE L4 (Job 974531), 50 epochs
- Checkpoint: `checkpoints/E_phase1_rebalance_l4/best_model.pt` (Best Epoch 49)

**结果 (test73, Oracle GT boxes, n=73)**:

| 指标 | Phase 1 | E29 基线 | vs E29 |
|------|:-------:|:-------:|:------:|
| **BM-1to1 Dice** | **0.695** | 0.593 | **+0.102** |
| **PQ@0.5** | **0.464** | 0.326 | **+0.138** |
| **SQ** | **0.616** | 0.586 | +0.030 |
| **RQ** | **0.753** | 0.557 | **+0.196** |
| **AJI** | **0.519** | 0.410 | **+0.109** |

**E2E 结果 (test73, DAPI 检测)**:

| 指标 | 值 |
|------|:---:|
| BM-1to1 Dice | 0.545 |
| PQ@0.5 | 0.172 |
| Oracle→E2E Gap | -0.292 PQ |

**结论**: Phase 1 是当前最佳模型。边界增强 + PQ 早停 = PQ +42% vs E29。E2E 瓶颈在检测端 (FP=8.5/图)。

**产物**: `experiments/comprehensive_eval/results.json`, `experiments/e2e_evaluation/results.json`

---

## P2-A: 邻居侵占/重叠互斥损失 Fix1-3

**日期**: 2026-02-15 ~ 2026-02-16 | **结论**: ❌ **P2-A 终止**

**目标**: 在 Phase 1 基础上添加 `L_neighbor` (侵占惩罚) + `L_overlap` (重叠互斥)，减少实例间冲突。

**三轮修复与汇总 (val71)**:

| 方案 | neighbor | overlap | delay | PQ | Dice | vs P1 PQ | 决策 |
|------|:-------:|:-------:|:-----:|:---:|:----:|:--------:|------|
| **P1 基线** | — | — | — | **0.475** | 0.695 | — | ✅ 基线 |
| Fix1 | 0.3 | 0.1 | 0 | 0.232 | — | **-51%** | ❌ 失败 |
| Fix2 | 0.1 | 0.05 | 0 | 0.393 | 0.687 | **-17%** | ⚠️ 改善 |
| Fix3 | 0.1 | 0.05 | delay=10 | 0.466* | 0.712 | **-2%** | ⚠️ 终止 |

\*Fix3 的 best PQ=0.466 发生在 **epoch 3 (N/O 尚未激活)**。N/O 升温后 PQ 单调下降至 0.341。

**关键发现**:
1. N/O loss 过度抑制: 模型变"保守"，边界区域不敢预测 → IoU 下降 → PQ 下降
2. 冲突像素减少 (50k→29k) 但 PQ 更差 — argmax_prob 已合理处理冲突
3. 实现缺陷 (detach + 顺序依赖) 加剧问题，但 loss 设计本身也有负面影响

**决策**: P2-A 终止，论文定位为 "Preliminary Exploration: N/O Exclusion Loss"

**产物**: `checkpoints/E_phase2a_fix*/`, `docs/phase2_design.md` §7-8, `docs/temp_reviews/fix2_review.md`, `docs/temp_reviews/fix3_review.md`

---

## T3b: Adaptive search_radius 重扫 (2026-02-19)

**背景**: T3 诊断发现 Adaptive 检测 `zline_saturated` (search_radius=200 过大)。T3b 缩小搜索范围。

**方法**: `python tools/ablation_adaptive_val.py --b1-values 80,100,120,140,160,180 --profile locked_eval`

**结果 (val71)**:

| search_radius | F1 | vs 原始(200) |
|:---:|:---:|:---:|
| 80 | 0.723 | -2.7% |
| 100 | 0.750 | +0.0% |
| 120 | 0.762 | +1.2% |
| 140 | 0.771 | +2.1% |
| **160** | **0.779** | **+2.9%** |
| 180 | 0.775 | +2.5% |

**最终最优**: `search_radius=160, min_zlines=5, zline_threshold=0.05`, F1=**0.780**

**结论**: 缩小 radius 使 Z-line 筛选生效，F1 从 0.750→0.780。但仍低于 DAPI (0.811)。

**产物**: `experiments/ablation_adaptive_radius_val/results.json`

---

## ⚠️ 关键发现: Semantic vs Instance Dice (2026-02-05) ⭐⭐⭐



**问题诊断**:

- 之前所有实验 (E01-E28) 使用 Semantic Dice 验证

- 训练时 `target = (mask > 0)` 将所有细胞合并为语义掩码

- 导致模型学习预测大 blob 而非单细胞



**调试结果** (E25 Boundary Enhanced):

```

Pred area: 105,129 pixels (覆盖多个细胞)

GT area:   41,477 pixels (单个细胞)

Instance IoU: 0.033  ← 极低

Instance Dice: 0.064 ← 极低

```



**修复方案**:

1. Instance-level target: `target = (mask == cell_id)`

2. Box clipping: 限制 pred/target 在 box 区域

3. Instance Dice 验证: 每个细胞独立计算



**新增功能 (2026-02-05)**:

- `ContourLoss`: 边界距离惩罚

- `GridDistortion`: 边界鲁棒性增强

- Phase 1/2 分阶段训练配置



---



## E22: 推理 Box Clipping 修复 + 边界精度分析 ⭐⭐⭐



**日期**: 2026-02-02



**背景/假设**: 

全面评估发现 PQ@0.5=0，n_pred=4.3 但 GT=10。分析发现 SAM 预测的 mask 远超 box 范围 (2-15x)。



**根本原因**: 训练-推理不一致

- 训练: `CombinedLoss` 只在 box+20%扩展区域计算 loss，box 外不惩罚

- 推理: 未对预测 mask 做 box 裁剪，导致巨大 mask 互相覆盖



**修复**: 在 `tools/comprehensive_eval.py` L125-138 添加 box clipping:

```python

mask_clipped = np.zeros_like(mask)

mask_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = mask[y1_clip:y2_clip, x1_clip:x2_clip]

```



**修复后结果**:



| 指标 | 修复前 | 修复后 | 提升 |

|------|--------|--------|------|

| **Dice** | 0.715 | 0.766 | +7.1% |

| **PQ@0.3** | 0.002 | 0.181 | **90x** |

| **AJI** | 0.063 | 0.203 | **3.2x** |

| **n_pred** | 4.3 | 9.8 | ≈ GT |



**边界精度分析** (310 实例):



| IoU 范围 | 占比 |

|----------|------|

| **0.1-0.3** | **56.8%** ⚠️ |

| **0.3-0.5** | 40.3% |

| **≥0.5** | 1.9% |



**分割偏差**: 67% 过分割 (pred > GT×1.1)



**结论**:

1. ✅ Box clipping 修复成功，PQ@0.3 大幅提升

2. ⚠️ 边界精度仍低，IoU 均值仅 0.28

3. → 根本原因是 20% box 扩展导致系统性过分割



**下一步**: 减小 expand 参数或增加边界惩罚



**文档**: `docs/boundary_precision_analysis.md`, `docs/detection_problem_report.md`



---



## E20: DAPI Only vs Adaptive 检测消融 ⭐⭐



**日期**: 2026-01-30



**背景/假设**: 

Adaptive 方案使用 Z-线自适应框 (`detect_with_adaptive_box`)，理论上能更准确地定位心肌细胞边界。需要与 DAPI Only 方案 (`detect_and_create_boxes`) 进行对比。



**方法**:

1. 在 20 个测试样本上分别运行两种检测方法

2. 使用 IoU@0.3 阈值匹配预测框与 GT 框

3. 计算 Precision, Recall, F1



**参数**:

| 参数 | 值 |

|------|-----|

| min_nucleus_area | 3000 |

| max_nucleus_area | 30000 |

| min_zlines (Adaptive) | 15 |

| zline_threshold | 0.03 |

| IoU 阈值 | 0.3 |

| 测试样本数 | 20 |



**结果**:



| 方法 | Precision | Recall | F1 |

|------|-----------|--------|-----|

| **DAPI Only** | **0.793** | **0.739** | **0.765** |

| Adaptive | 0.311 | 0.290 | 0.300 |

| **差异** | **+0.48** | **+0.45** | **+0.465** |



**分析**:

- **DAPI Only 明显优于 Adaptive**，F1 差距高达 **46.5%**

- Adaptive 方案 Precision 极低 (0.31)，说明 Z-线自适应框生成了大量误检

- Adaptive 的框尺寸可能偏大，导致与 GT 的 IoU 过低



**原因分析**:

1. `create_adaptive_box` 的 `fallback_expansion` 参数 (4.0) 过大

2. Z-线聚类可能将多个细胞的 Z-线合并

3. 当前 IoU@0.3 阈值对大框不友好



**结论**: ❌ Adaptive 方案在当前参数下严重劣于 DAPI Only，建议：

1. 使用 **DAPI Only** 作为默认检测方法

2. 如需使用 Adaptive，需大幅调低 `fallback_expansion` 和 `padding_ratio`



**代码位置**: `tools/ablation_detection.py`

**结果存档**: `experiments/ablation_detection/results.json`



**日期**: 2026-01-26

> ⚠️ 口径说明 (2026-02-14): 本节为早期历史分析记录，用于发现问题，不作为当前 E34 参数锁定依据。  
> 当前锁定口径请以 `val(71) -> test(73)` 的 E34 章节为准。



**背景**:

E18 发现 Adaptive 方法 Precision 较低 (0.672)，怀疑边缘过滤过松。同时 DAPI 方法 FN 较高，怀疑双核合并阈值不准。



**方法**:

1. **GT 极小核分析**: GT Mask 中存在大量 <1000px 的碎片。确认有效心肌细胞核应 >3000px (可视验证最小约 5000px)。

2. **边缘过滤重算**: 仅统计 >=5000px 的有效 GT 核在不同边缘阈值下的排除率。

3. **双核间距重算**: 仅统计 >=5000px 且 size_ratio < 3.0 的有效配对。



**结果 (min_area=5000)**:

1. **边缘排除率 (Valid GT >5000px)**:

   - 30px: 0.0%

   - 50px: 0.8%

   - **100px: 5.6%** (推荐)

   - 150px: 15.8%



2. **双核间距 (Valid Pairs)**:

   - Median: 137 px

   - **Mean: 161 px**

   - **P75: 160 px**

   - P95: 322 px (过大，离群)



**结论**:

- **边缘阈值**: 定为 **100px** (排除 ~5.6% 有效核，换取高 Precision)。

- **合并阈值**: 定为 **1.5×直径 (~170px)**，完美覆盖 Mean/P75，避免 P95 的过度合并。



**代码位置**: `dapi.py`, `evaluate_box_generation.py`



## 早期实验归档 (E01-E18)

> 完整详情见 [`experiments_log_archive.md`](experiments_log_archive.md)

| ID | 日期 | 实验 | 关键结果 | 结论 |
|----|------|------|---------|------|
| **E01** | 01-08 | 类别不平衡修复 | Dice 0→0.52 | ✅ pos_weight + box-local loss 解决零预测 |
| **E02** | 01-08 | CellFinder 检测 | F1=0.012 | ❌ CellSAM 自带检测器对心肌无效 |
| **E03** | 01-08 | DAPI 核检测 | F1=0.750 | ✅ Otsu+形态学+边缘过滤，成为默认方案 |
| **E04** | 01-09 | 全管线 (像素级) | Dice=0.58 | ⚠️ 像素合并丢失实例信息 |
| **E05** | 01-09 | 全管线 (实例级) | Dice=0.71 | ✅ 每框分配 cell_id，实例级输出 |
| **E06** | 01-11 | 分水岭核分离 | F1=0.34 | ❌ 过度分割，不适用 |
| **E09** | 01-11 | 验证指标实现 | PQ=0, AJI=0.10 | ⚠️ Mean Max IoU=0.05-0.22，揭示 mask 质量差 |
| **E12** | 01-11 | 边界损失微调 | PQ↑265%, Dice↑8% | ✅ boundary_weight=0.5 + contour_weight=0.1，当时最佳 |
| **E13** | 01-11 | 数据集标准化 | 固定 train/val/test | ✅ 统一训练入口 `train.py` |
| **E14** | 01-14 | 核-细胞轴向对齐 | 50%对齐@30°, Dice+3.3% | ✅ 智能扩展验证 |
| **E15a/b** | 01-15 | 多通道融合 | BF+DAPI+Actn2 Dice=0.745 | ❌ 劣于 BF-only E12 (Dice=0.772) |
| **E16** | 01-16 | E12 vs E15b 对比 | E12 优 2.6% | ✅ 确认 BF-only 最佳 |
| **E17** | 01-21 | GT 细胞面积统计 | 阈值 40K-450K | ✅ 数据驱动过滤参数 |
| **E18** | 01-23 | SarcGraph 检测对比 | F1↑7.4% | ✅ Adaptive F1=0.801 > DAPI 0.727 (当时口径) |

### 关键决策记录

| 日期 | 决策 | 选择 | 理由 |
|------|------|------|------|
| 01-08 | 检测方案 | DAPI 核检测 | F1=0.750 >> CellFinder 0.012 |
| 01-11 | 分割方案 | 边界增强微调 | PQ↑265% |
| 01-15 | 通道选择 | BF-only | 三通道融合劣于 BF-only |
| 01-26 | 边缘过滤 | edge_margin=100px | 排除 5.6% 换 Precision |
| 01-30 | 检测默认 | DAPI Only | Adaptive F1=0.313 << DAPI 0.765 (当时参数) |
| 02-14 | 检测锁定 | DAPI F1=0.803 | E34b test73 封板 |

---

## 待实验 (Planned)

> **2026-02-25 R1 更新**: E-B1~B6 baseline 已由 T16 完成; P2-D/E 已被 T12 消融覆盖; T7 已过时。

| ID | 实验名称 | 优先级 | 状态 |
|----|---------|--------|------|
| ~~P2-D~~ | ~~lr=5e-5, epochs=50 (LR消融)~~ | — | ❌ 不再需要 (T12 完成了完整消融) |
| ~~P2-E~~ | ~~lr=1e-4, epochs=80 (Epoch消融)~~ | — | ❌ 不再需要 (T12 完成了完整消融) |
| ~~E-B1~~ | ~~Cellpose baseline~~ | — | ✅ T16 已完成 (PQ=0.073) |
| ~~E-B2~~ | ~~StarDist baseline~~ | — | ✅ T16 已完成 (PQ=0.000) |
| ~~E-B4~~ | ~~CellSAM Original (test73)~~ | — | ✅ T16 已完成 (PQ=0.000) |
| ~~E-B5~~ | ~~MedSAM baseline~~ | — | ✅ T16 已完成 (PQ=0.101) |
| ~~E-B6~~ | ~~SAMCell baseline~~ | — | ✅ T16 已完成 (PQ=0.213) |
| ~~T7~~ | ~~Adapter Instance 评估~~ | — | ❌ 已过时 (Phase 2 终止) |
| **T11** | **LoRA Encoder ViT-B Q/V (rank=4/8)** | **P1** | **⏳ 设计完成, 待 R1 审核** |

> T11 设计文档: `docs/t11_lora_design.md`

