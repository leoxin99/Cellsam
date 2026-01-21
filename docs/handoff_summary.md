# CellSAM 项目交接摘要

> **用途**: 新 AI 对话窗口快速入门
> **日期**: 2026-01-21
> **状态**: 阶段2 模型训练 (80%)

---

## 快速了解项目

**新窗口必读顺序**:
1. `CLAUDE.md` - 项目总览 (5分钟)
2. `anti_test/experiments_log.md` - 实验历史 (按需查阅)

**按需查阅**:
- `docs/design_decisions.md` - 设计决策理论
- `docs/technical_details.md` - 技术规格
- `docs/troubleshooting.md` - 常见问题

---

## 项目一句话概述

**CellSAM**: 将 SAM (Segment Anything Model) 微调用于 hiPSC-CM (心肌细胞) 自动分割。

---

## 当前状态

| 项目 | 状态 |
|------|------|
| **最佳模型** | E12 (Pixel Dice 0.7718) |
| **输入** | BF×3 (明场3通道复制) |
| **检测方法** | DAPI 核检测 (替代 CellFinder) |
| **数据集** | Allen Cell, 478 图, 5173 细胞 |

---

## 今日完成 (2026-01-21)

### 新建代码模块
- `src/inference/` - 统一推理 (平滑、着色、大小验证)
- `src/detection/dapi.py` - DAPI核检测 + 智能双核合并
- `tools/run_inference.py` - 统一推理脚本

### 数据分析
- E17: 全数据集 GT 统计 (5173 细胞)
- 阈值: P1=40836, P99=513928

### 文档优化
- CLAUDE.md 从 767 行精简到 ~170 行
- 创建 3 个详细文档 (design, technical, troubleshooting)

### 问题解决
- 问题2: 双核距离智能合并
- 问题3: 6步边界平滑
- 问题4.1: 图着色 (相邻不同色)
- 问题4.2: 细胞大小阈值

---

## 待处理任务

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | SizeLoss 集成到 CombinedLoss | 待实现 |
| P1 | 完整 478 样本训练 | 待执行 |
| P2 | 测试集完整评估 | 待执行 |

---

## 关键决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 训练时用什么框 | GT 框 | 解耦训练 |
| 推理时用什么检测 | DAPI | CellFinder 对心肌细胞失效 |
| 多通道输入 | ❌ 放弃 | E15b 失败 |
| 冻结策略 | 仅训练 Decoder | 效率+防过拟合 |

---

## 代码入口点

```python
# 推理
from inference import run_sam_inference, mask_to_rgb
from detection import detect_and_create_boxes

boxes = detect_and_create_boxes(dapi_image)
pred = run_sam_inference(model, bf_image, boxes)

# 训练
python src/train.py --config src/config/base.yaml
```

---

## 实验编号参考

| ID | 重要实验 |
|----|----------|
| E12 | ⭐ 边界损失微调 (当前最佳) |
| E15b | ❌ 多通道失败 |
| E17 | GT 细胞面积统计 |

完整实验: `anti_test/experiments_log.md`

---

*新窗口只需阅读 CLAUDE.md 即可开始工作*
