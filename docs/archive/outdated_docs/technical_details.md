# CellSAM 技术规格

> **文档类型**: 技术参考文档 (从 CLAUDE.md 提取)
> **最后更新**: 2026-01-21

---

## 代码架构

```
src/
├── inference/              # 统一推理模块
│   ├── __init__.py
│   ├── postprocess.py     # 6步边界平滑 + 大小验证
│   ├── visualize.py       # 图着色 (4-color theorem)
│   └── pipeline.py        # run_sam_inference() 统一入口
├── detection/             # DAPI 核检测模块
│   ├── __init__.py
│   └── dapi.py            # 核检测 + 智能双核合并
├── losses/                # 损失函数
│   └── combined.py        # Dice+BCE+Boundary+AJI+SizeLoss
├── config/                # 配置文件
│   ├── base.yaml
│   └── boundary.yaml
└── train.py               # 主训练入口

tools/
├── run_inference.py       # 统一推理脚本
└── view_minmax_cells.py   # 调试工具

anti_test/
├── visualize_test_results.py  # 历史参考
└── experiments_log.md         # 实验记录
```

---

## 模型参数

| 组件 | 参数量 | 状态 |
|------|--------|------|
| Image Encoder (ViT-H) | 630M | ❄️ 冻结 |
| Prompt Encoder | - | ❄️ 冻结 |
| Mask Decoder | 4M | 🔥 训练 |
| CellFinder | - | ❄️ 冻结 |

---

## 损失函数

| 损失 | 权重 | 用途 |
|------|------|------|
| DiceLoss | 0.5 | 基础分割 |
| BCELoss | 0.5 | 二分类 |
| BoundaryLoss | 0.3 | 边界精度 |
| AJILoss | 0.2 | 实例级质量 |
| SizeLoss | 待集成 | 大小约束 |

---

## 数据集统计 (Allen Cell)

| 指标 | 值 |
|------|-----|
| 图片数 | 478 |
| 细胞数 | 5173 |
| 细胞面积 (P1) | 40,836 像素 |
| 细胞面积 (Median) | 142,316 像素 |
| 细胞面积 (P99) | 513,928 像素 |

**大小阈值**:
- MIN_CELL_AREA = 40,836 (P1)
- MAX_CELL_AREA = 513,928 (P99)

---

## 常用命令

### 训练
```bash
python src/train.py --config src/config/base.yaml
```

### 推理
```bash
python tools/run_inference.py --checkpoint checkpoints/e12_boundary_best.pt --samples 10
```

### 数据处理
```bash
python data/scripts/extract_expanded_pairs.py --limit 50
python data/scripts/generate_splits.py
```

---

## TIFF 通道映射 (已验证 2026-02-06)

| 通道 | 内容 |
|------|------|
| Ch0 | Brightfield (明场) |
| Ch1 | **Actn2 (肌动蛋白)** |
| Ch4 | DAPI (核染色) |
| Ch8 | Binary mask (二值掩码) |
| Ch9 | GT Segmentation Mask (实例分割) |

---

*详细设计决策请参考 docs/design_decisions.md*
