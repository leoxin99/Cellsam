# Codex Step 5-7 审核报告

> 审核日期: 2026-02-14  
> 审核人: Claude (Antigravity)

---

## ✅ Step 5: 脚本与参数入口

### `merge_coeff` 入口打通 — ✅ 正确
`dapi.py` 三个入口均已正确添加 `merge_coeff` 参数:

| 函数 | 行号 | 默认值 | 传递路径 |
|------|------|--------|----------|
| `src/detection/dapi.py:71` (`merge_close_nuclei`) | L71 | 1.2 | → L124-125 计算 `max_merge_dist` |
| `src/detection/dapi.py:231` (`detect_and_create_boxes`) | L231 | 1.2 | → L264 传入 `merge_close_nuclei` |
| `src/detection/dapi.py:539` (`detect_with_adaptive_box`) | L539 | 1.2 | → L586 传入 `merge_close_nuclei` |

> 所有入口参数签名一致，默认值均为 1.2 (历史经验值)，通过 kwarg 可覆盖。

### 新增脚本 — ✅ 结构正确

| 脚本 | 行数 | 功能 |
|------|------|------|
| `tools/ablation_detection_e34b.py` | 326 | val71 联合消融 (edge_margin × size_ratio × merge_coeff) |
| `tools/ablation_detection_lock.py` | 301 | test73 单次封板 (DAPI vs Adaptive) |

两个脚本都包含完整的:
- 数据加载 → GT 框提取 → IoU 匹配 → P/R/F1 计算
- CLI 参数入口 + 结果 JSON 写盘
- 策略标记 (`single_run_lockdown_no_reverse_tuning`)

---

## ✅ Step 6: E34b 联合消融 (val71)

### 搜索空间覆盖 — ✅ 完整
- `edge_margin`: [20, 32, 50] ✅
- `size_ratio_threshold`: [2.0, 2.5, 3.0, 3.5] ✅
- `merge_coeff`: [1.0, 1.2, 1.4, 1.5] ✅
- **共 3×4×4 = 48 组合** ✅ (results.json 含 48 条)

### 最优参数 — ✅ 可信

| 排名 | edge_margin | size_ratio | merge_coeff | P | R | F1 |
|------|-------------|------------|-------------|-------|-------|--------|
| **#1** | 20 | 2.5 | 1.4 | 0.7639 | 0.8633 | **0.8106** |
| #2 | 20 | 3.0 | 1.4 | 0.7639 | 0.8633 | 0.8106 |

### 深度分析与发现

> [!IMPORTANT]
> **`size_ratio_threshold` 在当前 val71 + 当前搜索空间下 (>=2.5) 不敏感**
> #1 和 #2 的 TP/FP/FN 完全相同 (644/199/102)，说明实际被 `size_ratio_threshold` 过滤的候选对为 0。这从逻辑上合理：心肌细胞双核通常大小接近，ratio>2.5 的对本就极少。

**参数敏感性排序**:
1. **`edge_margin`** — **高敏感**: 50→20 提升 F1 约 +5pp (0.76→0.81)，主要通过恢复边缘区域 recall
2. **`merge_coeff`** — **中等敏感**: 影响 F1 约 ±1pp，控制 P/R 权衡
3. **`size_ratio_threshold`** — **当前口径下不敏感**: >=2.5 后结果不变

> [!WARNING]
> **`edge_margin=20` vs 历史值 50 的含义**: 减小 margin 意味着保留更多边缘细胞，recall 大幅提升 (+11pp)，但 precision 基本不变。这是合理的，因为心肌细胞贴壁培养后边缘附近仍有完整细胞。但需注意 test 集上边缘分布是否一致。

---

## ✅ Step 7: Test73 封板

### 策略合规性 — ✅ 完整

| 检查项 | 状态 |
|--------|------|
| `policy = single_run_lockdown_no_reverse_tuning` | ✅ |
| 仅执行一次，无反向调参 | ✅ |
| DAPI 参数与 val 锁定一致 | ✅ (`1500/20000/20/2.5/1.4/relative`) |
| Adaptive 参数与 val 锁定一致 | ✅ (`radius=200/min_zlines=5/zline_threshold=0.01`) |

### 结果

| 检测器 | P | R | F1 | 状态 |
|--------|-------|-------|--------|------|
| **DAPI** | 0.7462 | 0.8699 | **0.8033** | 🏆 Winner |
| Adaptive | 0.6968 | 0.8123 | 0.7502 | |
| **Δ F1** | | | **-0.0531** | DAPI 胜出 |

### val→test 泛化分析

| 指标 | val(E34b) | test(lock) | Δ |
|------|-----------|------------|---|
| DAPI F1 | 0.8106 | 0.8033 | **-0.0073** |
| DAPI P | 0.7639 | 0.7462 | -0.0177 |
| DAPI R | 0.8633 | 0.8699 | +0.0066 |

> [!TIP]
> **DAPI val→test F1 仅下降 0.73pp，在当前同分布口径下泛化稳定。** Recall 略有上升，Precision 小幅下降可接受。

---

## 回归检查 — ✅

`python tools/test_unified_regression.py` → **10 passed, 0 failed**

---

## 可复现实验元信息

- commit: `f865e64`
- E34b 命令: `python -B -X utf8 tools/ablation_detection_e34b.py --split val`
- test 封板命令: `python -B -X utf8 tools/ablation_detection_lock.py --split test`
- 结果文件:
  - `experiments/ablation_detection_e34b/results.json`
  - `experiments/ablation_detection_lock/results.json`

---

## 📋 审核结论与下一步建议

### 结论: Codex 交付合格 ✅

1. **代码质量**: `merge_coeff` 入口正确打通，脚本结构清晰
2. **实验质量**: 搜索空间完整，结果可复现（有 JSON 记录）
3. **封板策略**: 严格执行 `single_run_lockdown`，无 test 泄漏
4. **回归**: 10/10 通过

### 建议下一步执行

根据 `task_backlog.md` 当前状态，建议执行以下更新:

1. **更新 `task_backlog.md`**:
   - T1 (E34 test 封板) → ✅ 完成
   - T2 (E34b 联合消融) → ✅ 完成
   - 回填 `dapi_detection_design.md`, `experiments_log.md`, `claude.md`

2. **更新 `claude.md`**:
   - Step 4.5 → ✅ 完成 (参数已封板)
   - Step 4.6 → ✅ 完成
   - 关键指标: DAPI(test) F1=0.8033
   - 关键决策表: 检测参数锁定值

3. **更新 `dapi_detection_design.md`**:
   - 标记 "参数已封板"
   - 写入最终锁定参数

4. **更新 `experiments_log.md`**:
   - 新增 E34b + E34-lock 记录

> [!IMPORTANT]
> **`size_ratio_threshold` 建议锁定为 2.5 而非更大值**，虽然 ≥2.5 后结果完全相同，但 2.5 是 safety margin 最合理的选择（避免极端大小比的误合并）。

需要我执行这些文档更新吗？
