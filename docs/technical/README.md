# 技术文档索引

> **状态**: 🟢 Active
> **最后更新**: 2026-03-07
> **用途**: 统一收纳项目技术分析、CellSAM 对照、训练策略问答与专题审计文档

---

## 1. 使用原则

- 本目录存放“技术分析/技术问答/论文证据整理”类文档。
- 运行口径、数据参数、检测参数等 SSOT 仍在顶层文档:
  - `docs/inference_standard.md`
  - `docs/dataset_parameters.md`
  - `docs/dapi_detection_design.md`
  - `docs/task_backlog.md`
  - `docs/experiments_log.md`
- 论文写作优先引用本目录文档，再回到代码或论文原文核对证据边界。

---

## 2. 文档清单

| 文档 | 用途 |
|------|------|
| `docs/technical/update_cellsam.md` | CellSAM 两阶段训练、loss 边界、与本项目差异总汇总 |
| `docs/technical/technical_qa_2.27.md` | 高频技术问答汇总 |
| `docs/technical/cellsam_ours_com_2.28.md` | CellSAM 官方流程 vs 我们项目流程逐模块对照 |
| `docs/technical/cellsam_methods_1page_table.md` | 论文可直接引用的一页式 methods 证据表 |
| `docs/technical/adapter_cellsam_tech_reference.md` | Adapter 设计、实现、训练集成与 CellSAM 数据口径 |
| `docs/technical/cellsam_sam_branch_audit_2026-02-21.md` | CellSAM SAM 分支专项审计 |
| `docs/technical/cellsam_update_predict_2.28.md` | CellSAM 预处理/推理链更新记录 |
| `docs/technical/question.md` | 历史项目技术问答整理 |
| `docs/technical/three_channel_design_evaluation.md` | 三通道设计与评估专题 |
| `docs/technical/adapter_analysis.md` | Adapter 早期分析记录 |
| `docs/technical/metrics_guide.md` | 指标说明与口径补充 |
| `docs/technical/cellpose_builtin_models_reference.md` | Cellpose built-in models、适用域与心肌细胞适配性说明 |

---

## 3. 建议阅读顺序

1. `docs/technical/update_cellsam.md`
2. `docs/technical/cellsam_ours_com_2.28.md`
3. `docs/technical/technical_qa_2.27.md`
4. `docs/technical/adapter_cellsam_tech_reference.md`
5. `docs/technical/cellsam_methods_1page_table.md`
6. `docs/technical/cellpose_builtin_models_reference.md`

---

## 4. 维护规则

- 新增技术分析类文档时，优先放入 `docs/technical/`。
- 若顶层文档中只需引用技术结论，不复制整段表格，直接链接到本目录对应文档。
- 历史文档若升级为 SSOT，应迁回顶层；否则继续保留在本目录或归档。
