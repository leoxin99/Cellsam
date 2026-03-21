# 公式与引用状态审计（2026-03-20）

> 作者: A3 (Codex)  
> 用途: 第一性原理下，明确“论文该有多少公式、现在缺什么、引用还缺什么”

## 1. 当前状态（结论先行）

1. 目前正文对原理的解释以文字和表格为主，**公式密度偏低**。  
2. 指标关系（如 `PQ = SQ x RQ`）已出现，但 loss 与检测目标缺少标准化公式表达。  
3. `Markdown -> LaTeX` 新导出产物（`overleaf_export/chapters/*.tex`）目前几乎没有 `\cite{}`，说明**引用尚未完成 pass**。  

## 2. 同类论文常见公式设计（建议）

通常保持“少而硬”的公式策略，正文放 6~10 个核心公式即可：

1. **任务与输出定义**（实例集合、prompt 条件）。
2. **训练目标**（主 loss + IoU-head / 辅助项）。
3. **检测目标（若写 CellFinder）**：分类 + box 回归 + GIoU。
4. **核心评估指标**：PQ、SQ、RQ、F1 的关系。
5. **可选**：后处理或匹配规则（若对结论关键）。  

不建议把实现细节公式化到过深层级（例如每个模块内部算子都展开）。

## 3. 建议补入的最小公式包（按章节）

## 3.1 Chapter 3（评估）

1. `PQ = SQ * RQ`
2. `F1 = 2TP / (2TP + FP + FN)`
3. `RQ` 与 `F1` 的口径说明（同一 `TP/FP/FN` 定义下等价）

## 3.2 Chapter 4（方法/训练）

1. `L_total = L_combined + lambda_iou * L_iou`
2. `L_combined` 的加权展开（Dice/BCE, Boundary, AJI, Focal）
3. `L_iou = MSE(q_pred, q_target)`

## 3.3 Chapter 4 或 Appendix（检测）

若正文保留 CellFinder 机制图，建议补一条：

`L_det = L_cls + lambda_L1 * L_box + lambda_giou * L_giou`

## 4. 引用整理状态

1. `references.bib` 已有核心条目（SAM / CellSAM / MedSAM / Cellpose / LoRA / PQ / AJI 等）。  
2. `prism_import/chapters/*.tex` 历史版本里有较完整引用。  
3. 新导出的 `overleaf_export/chapters/*.tex` 来自 Markdown，当前基本不含 `\cite{}`。  

结论：需要做一次系统 citation pass，把章节中的关键论断补上 citation key。

## 5. 需要 A1 / A2 补充的信息（用于收口）

## 5.1 需要 A1 提供

1. H1b 封板后的最终可写口径（最终采用的 detector 分支、阈值、协议）。  
2. 对应图文件清单（可直接放论文的最终路径 + 样本编号）。  
3. 训练与推理脚本版本锚点（commit 或 patch 编号）。  

## 5.2 需要 A2 提供

1. `T31` Cellpose v4 主表最终口径（版本、直径、汇总指标、是否还有补跑）。  
2. `T28` 的最终“可写主口径”指标包（建议含 `test73` 全指标与 per-sample JSON）。  
3. Adapter 结论封板说明（用于正文一句话结论 + 附录是否保留）。  

## 5.3 需要 A1/A2 一致确认

1. 主表中的 split 和协议一致性（`test73` vs `val71` 不混写）。  
2. Oracle 与 E2E 图/表的术语边界。  
3. 最终 baseline 版本统一（当前口径：Cellpose v4）。  

