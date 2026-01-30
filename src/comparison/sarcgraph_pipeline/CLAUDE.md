# CellSAM 心肌细胞分割项目

## 项目背景

这是一个针对人诱导多能干细胞分化心肌细胞（hiPSC-CMs）的自动化分割方案，基于SAM模型进行优化。

## 核心问题

1. 原有的 cellfinder 无法识别心肌细胞
2. 自定义训练的检测算法效果不理想
3. 需要优化三通道输入策略

## 解决方案

### 方法一：数据层面 - 语义通道映射
- 文件：`preprocessing.py`
- 核心类：`SemanticChannelMapper`
- 通道映射：
  - R通道 ← α-actinin（肌节纹理）
  - G通道 ← Phase/明场（细胞边界）
  - B通道 ← DAPI（细胞核定位）

### 方法二：模型层面 - 通道适配器
- 文件：`channel_adapter.py`
- 三种适配器：
  - `LightweightChannelAdapter`：6个参数，快速验证
  - `IndependentChannelAdapter`：30个参数，推荐使用
  - `ICViTStyleAdapter`：~15000参数，大数据集使用

### 提示框生成：SarcGraph驱动
- 文件：`prompt_generator.py`
- 流程：Z线检测（LoG） → DBSCAN聚类 → 边界框生成
- 核心类：`SarcGraphPromptGenerator`

### 完整Pipeline
- 文件：`pipeline.py`
- 核心类：`CellSAMPipeline`, `CellSAMTrainer`

## 当前任务

继续完善以下方面：
1. 与实际SAM模型的集成
2. 训练循环的完整实现
3. 数据加载器
4. 评估指标（IoU, Dice, 肌节密度验证）
5. 可视化工具

## 技术栈

- Python 3.8+
- PyTorch 2.0+
- segment-anything
- scikit-learn (DBSCAN)
- scikit-image (blob detection)
- OpenCV (CLAHE)

## 关键参数

```python
# DBSCAN参数
eps_pixels = (sarcomere_length_um * eps_factor) / pixel_size_um
# 典型值：(2.0 * 2.0) / 0.5 = 8 像素

# 预处理参数
actn2_percentile = (0.5, 99.5)  # 截断最亮0.5%
phase_clahe_clip = 2.0
```

## 参考研究

- SAMCell (PLOS One, 2025)
- CellSAM (Nature Methods, 2025)
- μSAM (Nature Methods, 2024)
- Sarc-Graph (PLOS Computational Biology, 2021)
