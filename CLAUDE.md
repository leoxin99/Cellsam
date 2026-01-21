# CellSAM 项目方案 (Project Blueprint)

> **文档类型**: 项目总览 (AI 必读)
> **最后更新**: 2026-01-21
> **当前阶段**: 阶段2 - 边界优化

---

## 项目状态仪表板

### 整体进度
```
阶段1 数据准备   [████████████████████] 100%  ✅ 完成
阶段2 模型训练   [████████████████░░░░]  80%  🔄 边界平滑优化中
阶段3 评估验证   [████████░░░░░░░░░░░░]  40%  🔄 实例+像素评估完善
阶段4 论文结果   [░░░░░░░░░░░░░░░░░░░░]   0%  ⏳ 待开始
```

### 关键指标
| 指标 | 当前值 | 目标值 | 状态 |
|-----|-------|-------|------|
| **Detection F1** | **0.750** | 0.85+ | ✅ 良好 |
| **Pixel Dice (E12)** | **0.7718** | 0.85+ | ✅ 当前最佳 |
| **Instance Dice** | 待优化 | 0.7+ | 🔄 进行中 |

### 当前最佳模型
- **路径**: `checkpoints/boundary_20260111_012636/best_model.pt` (E12)
- **输入**: BF × 3
- **性能**: Pixel Dice 0.7718

---

## 代码架构

```
src/
├── inference/           # 统一推理模块 (NEW)
│   ├── postprocess.py  # 6步边界平滑 + 大小验证
│   ├── visualize.py    # 图着色
│   └── pipeline.py     # run_sam_inference()
├── detection/          # DAPI 检测模块 (NEW)
│   └── dapi.py         # 核检测 + 智能双核合并
├── losses/
│   └── combined.py     # Dice+BCE+Boundary+AJI+SizeLoss
└── train.py            # 主训练入口

tools/
└── run_inference.py    # 统一推理脚本 (NEW)
```

详细架构: [docs/technical_details.md](docs/technical_details.md)

---

## 阶段性任务清单

### 阶段1: 数据准备 ✅
- [x] 下载 Allen 数据集 (478 张 TIFF)
- [x] 验证通道映射 (Ch0=BF, Ch4=DAPI, Ch9=GT)
- [x] GT 统计分析 (E17: 5173 细胞, P1=40836, P99=513928)

### 阶段2: 模型训练 🔄
- [x] E12 边界损失微调 (当前最佳)
- [x] 统一推理管道
- [ ] 完整 478 样本训练

### 阶段3: 评估验证 ⏳
- [ ] 完整测试集评估
- [ ] 消融实验

---

## 关键决策速查

| 决策 | 选择 | 理由 | 详情 |
|------|------|------|------|
| 训练框 | GT 框 | 解耦训练 | [详情](docs/design_decisions.md#2) |
| 推理框 | DAPI 核检测 | CellFinder 失效 | [详情](docs/design_decisions.md#2) |
| 归一化 | P2-P98 | 鲁棒于异常值 | [详情](docs/design_decisions.md#3) |
| 冻结策略 | 仅训练 Decoder | 效率高，防过拟合 | [详情](docs/design_decisions.md#6) |
| 大小阈值 | P1/P99 | 排除标注错误 | [E17](anti_test/experiments_log.md) |

详细设计决策: [docs/design_decisions.md](docs/design_decisions.md)

---

## 文档架构

```
d:/AI/paper/CellSam/
├── CLAUDE.md                     # 📘 项目蓝图 (本文件) - AI 必读
├── docs/
│   ├── design_decisions.md       # 📐 设计决策详细理论
│   ├── technical_details.md      # 🔧 技术规格
│   └── troubleshooting.md        # � 常见问题解答
├── anti_test/
│   ├── experiments_log.md        # 📊 实验记录 (E01-E17)
│   └── methods_draft.md          # 📝 论文 Methods 草稿
└── src/                          # 源代码
```

| 文档 | 用途 | AI 操作 |
|------|------|--------|
| `CLAUDE.md` | 项目总览 | **必读** |
| `experiments_log.md` | 实验追溯 | 必须记录 |
| `design_decisions.md` | 论文参考 | 按需查阅 |
| `troubleshooting.md` | 问题解决 | 按需查阅 |

---

## AI 助手工作流程

1. **首先阅读** `CLAUDE.md` 了解项目状态
2. **查阅** `experiments_log.md` 了解已做实验
3. **执行实验** 并记录到 `experiments_log.md`
4. **更新** `CLAUDE.md` 状态仪表板 (有里程碑时)

---

## 更新日志 (最近5条)

| 日期 | 更新内容 |
|-----|---------|
| **2026-01-21** | **统一推理管道**, GT 统计 E17, 阈值 P1/P99, CLAUDE.md 优化 |
| 2026-01-16 | 废弃 E15b 多通道, AMP 混合精度 |
| 2026-01-14 | E14 智能扩展策略 +3.3% Dice |
| 2026-01-11 | E12 边界损失微调 PQ↑265% |
| 2026-01-09 | DAPI 检测替换 CellFinder |

完整日志: [anti_test/experiments_log.md](anti_test/experiments_log.md)

---

## 常见问题

| 问题 | 解决 | 详情 |
|------|------|------|
| Dice=0 | 边界框内损失 + pos_weight | [详情](docs/troubleshooting.md#q2) |
| 边界锯齿 | 6步平滑管道 | [详情](docs/troubleshooting.md#q5) |
| GPU OOM | batch_size=2, AMP | [详情](docs/troubleshooting.md#q3) |

完整 FAQ: [docs/troubleshooting.md](docs/troubleshooting.md)

---

*此文档由 AI 助手自动维护，每次重要进展后更新*
*详细内容请查阅链接文档*
