# CellSAM 心肌细胞分割项目进展汇报

**汇报日期**: 2025-12-30  
**项目目标**: 使用 CellSAM 模型对心肌细胞进行自动分割

---

## 一、项目进展

### 1.1 环境搭建 ✅ 已完成

| 项目 | 状态 |
|-----|------|
| Conda 环境配置 (Python 3.10) | ✅ |
| PyTorch + CUDA 安装 | ✅ |
| CellSAM 开发者模式安装 | ✅ |
| Napari 可视化插件 | ✅ |

### 1.2 数据获取 ✅ 已完成

| 数据集 | 数量 | 说明 |
|-------|------|------|
| Allen hiPSC-CM 标注数据 | 5 个 | 包含原图 + GT 分割掩膜 |
| Allen Brightfield 3D 原图 | 3 个 | 50 层 Z-stack (用于探索投影方法) |
| Cell Tracking Challenge 数据 | 2 个数据集 | 备用通用训练数据 (非心肌细胞) |

### 1.3 数据处理流程 ✅ 已完成

- **理解 Allen 数据结构**: 确认标注文件为 10 通道 OME-TIFF 格式
  - Ch 0-4: 原始图像 (Brightfield, GFP, DAPI 等)
  - Ch 6-9: Ground Truth 分割掩膜
- **投影方法分析**: 确认 Allen 使用 Max Projection (根据官方流程图)
- **编写预处理脚本**: `preprocess_allen_data.py` 支持 Sum/Max Projection

### 1.4 训练框架 ✅ 已完成

| 脚本 | 功能 |
|-----|------|
| `cardiomyocyte_dataset.py` | PyTorch 数据集类 |
| `train_cardiomyocyte.py` | 微调训练脚本 |
| `generate_prompts_from_dapi.py` | DAPI → 边界框 Prompt 生成 |

**训练策略**:
- 冻结 ViT Image Encoder (保留预训练权重)
- 只训练 Mask Decoder (适应心肌细胞形态)
- 使用 DAPI 核检测生成 Prompt

### 1.5 模型验证 ✅ 已完成

- 在下载的 Brightfield 图像上成功运行 CellSAM 推理
- 结果: 检测到 14 个细胞 (预处理后图像)

---

## 二、当前面临的问题

### 2.1 🔴 HPC 服务器访问受阻

| 问题 | 状态 |
|-----|------|
| eduVPN 连接 | ✅ 已解决 (需手动安装 Wintun 驱动) |
| SSH 连接 ALICE | ❌ 账号可能未激活 |

**错误信息**: `Connection closed by remote host` (SSH 握手阶段被拒绝)

**待办**: 联系 `helpdesk-alice@science.leidenuniv.nl` 确认账号状态

### 2.2 🟡 数据投影方法选择

| 问题 | 说明 |
|-----|------|
| Allen 使用 Max Projection | 可能丢失肌节纹理信息 |
| Sum Projection 可能更优 | 但需要重新对齐 GT 掩膜 |

**当前决策**: 先用 Allen 的 Max Projection 数据建立 Baseline，后续根据效果决定是否切换

### 2.3 🟡 数据量有限

| 数据集 | 数量 | 是否心肌细胞 |
|-------|------|------------|
| Allen 标注数据 | 5 个 | ✅ 是 |
| CTC 数据 | ~300 张 | ❌ 否 (胶质瘤/模拟) |

**解决方向**: 
- 使用伪标签扩展数据
- 申请更多 Allen 数据访问权限

---

## 三、下一步计划

| 优先级 | 任务 | 预计时间 |
|-------|------|---------|
| 高 | 解决 ALICE HPC 访问问题 | 1-2 天 |
| 高 | 使用 Allen 标注数据开始训练 | 1 周 |
| 中 | 评估 Max vs Sum Projection 效果 | 训练后 |
| 中 | 扩展训练数据量 | 根据需要 |

---

## 四、已产出的代码资产

```
d:\AI\paper\CellSam\
├── 数据处理
│   ├── preprocess_allen_data.py     # 3D→2D 投影
│   ├── generate_prompts_from_dapi.py # Prompt 生成
│   └── view_annotation_tiff.py      # 数据分析
├── 训练框架
│   ├── cardiomyocyte_dataset.py     # 数据集类
│   └── train_cardiomyocyte.py       # 训练脚本
├── 可视化
│   ├── view_with_cellsam.py         # Napari + 分割叠加
│   └── run_cellsam.py               # 推理脚本
└── 文档
    ├── PROJECT_SUMMARY.md           # 项目总结
    └── CellSAM_run_tutorial.md      # 使用教程
```

---

**汇报人**: [您的姓名]  
**导师**: [导师姓名]
