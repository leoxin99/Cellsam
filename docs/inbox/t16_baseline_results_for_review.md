# T16 Baseline 实验最终报告 — 请 R1 审核

**日期**: 2026-02-21
**状态**: 6/7 baseline 完成，仅 StarDist 待补

---

## 完整结果表

| Method | Type | PQ ↑ | BM-Dice ↑ | AJI ↑ | Sem.Dice | TP | FP | FN |
|--------|------|------|-----------|-------|----------|-----|-----|-----|
| Cellpose v4 | E2E | 0.000 | 0.053 | 0.025 | 0.079 | 0.0 | 255 | 10.0 |
| CellSAM Pretrained | Oracle | 0.000 | 0.121 | 0.056 | 0.219 | 0.0 | 10.0 | 10.0 |
| SAMCell (LIVECell) | E2E | 0.000 | 0.008 | 0.004 | 0.014 | 0.0 | 8.1 | 10.0 |
| SAM ViT-B (vanilla) | Oracle | 0.286 | 0.631 | 0.440 | 0.756 | 4.9 | 5.1 | 5.1 |
| **Ours Phase1_L4** | **Oracle** | **0.464** | **0.695** | **0.519** | **0.756** | **7.5** | **2.4** | **2.5** |
| **MedSAM** | **Oracle** | **0.576** | **0.771** | **0.634** | — | — | — | — |
| Ours Phase1_L4 | E2E | 0.180 | 0.567 | 0.338 | 0.642 | 3.5 | 8.2 | 6.5 |

> [!WARNING]
> **MedSAM Oracle (PQ=0.576) 超过了 Ours Oracle (PQ=0.464)**
> MedSAM (医学微调 SAM) 在相同 GT box 条件下表现更好，说明我们的分割微调策略还有提升空间。

## 关键发现

1. **Cellpose v4、SAMCell 完全失败 (PQ=0)** — iPSC-CM 太大太不规则，通用细胞分割模型无法处理
2. **CellSAM 未微调也失败 (PQ=0)** — 证明 domain-specific 微调不可或缺
3. **SAM ViT-B vs MedSAM**: MedSAM (PQ=0.576) >> SAM ViT-B (PQ=0.286) — 医学预训练有巨大价值
4. **MedSAM > Ours**：MedSAM 分割质量更高，但**无检测能力**——只能做 Oracle。我们的 E2E 管线是唯一能自动工作的方案
5. **E2E 瓶颈在检测**：Ours Oracle PQ=0.464 vs Ours E2E PQ=0.180，检测模块是关键瓶颈

## 待完成

- [ ] StarDist (需独立 TensorFlow conda 环境)

## 需要 R1 决策

1. **MedSAM > Ours Oracle** — 是否需要在论文中讨论此差距？建议方向：
   - 我们的微调聚焦于 iPSC-CM 特定任务，MedSAM 是百万级医学数据预训练
   - 我们提供了完整的 E2E pipeline（检测+分割），MedSAM 只有分割
2. **StarDist 优先级** — 是否需要补充？
3. **结果表是否足够支撑论文要求？**

## 文件索引
- MedSAM 结果: `experiments/baseline_comparison/results.json`
- SAMCell 结果: `experiments/baseline_comparison/per_sample_samcell_livecell.json`
- E2E 结果: `experiments/e2e_evaluation/results.json`
- Oracle 结果: `experiments/comprehensive_eval/results.json`
