# CellSAM Pipeline for Cardiomyocyte Segmentation

心肌细胞分割优化方案的完整实现，结合数据预处理（方法一）和模型适配（方法二）。

## 📁 文件结构

```
cellsam_pipeline/
├── preprocessing.py      # 方法一：数据层面的语义通道映射
├── channel_adapter.py    # 方法二：模型架构层面的通道适配器
├── prompt_generator.py   # SarcGraph驱动的提示框生成器
├── pipeline.py           # 完整的CellSAM Pipeline
├── requirements.txt      # 依赖包
└── README.md            # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用

```python
from pipeline import create_pipeline

# 创建Pipeline（不需要SAM权重即可测试）
pipeline = create_pipeline(
    adapter_type="independent",  # 推荐
    pixel_size_um=0.5
)

# 加载您的图像
import numpy as np
actinin = np.load("your_actinin.npy")  # α-actinin通道
phase = np.load("your_phase.npy")      # 相位差/明场
dapi = np.load("your_dapi.npy")        # DAPI（可选）

# 执行分割
result = pipeline.segment(actinin, phase, dapi)

# 查看结果
print(f"检测到 {result['num_cells']} 个细胞")
print(f"质量检查: {result['quality_check']}")

# 获取边界框
for box in result['boxes']:
    print(f"边界框: {box.to_xyxy()}")
```

### 3. 使用SAM进行实际分割

```python
from pipeline import CellSAMPipeline

# 创建带SAM的Pipeline
pipeline = CellSAMPipeline(
    sam_checkpoint="sam_vit_b.pth",  # SAM权重路径
    sam_model_type="vit_b",
    adapter_type="independent",
    pixel_size_um=0.5
)

# 执行分割
result = pipeline.segment(actinin, phase, dapi)

# 获取掩膜
masks = result['masks']  # List[np.ndarray]
```

## 📖 详细文档

### 方法一：数据预处理 (preprocessing.py)

将多通道显微镜图像转换为SAM友好的伪RGB格式。

```python
from preprocessing import SemanticChannelMapper

mapper = SemanticChannelMapper(
    actn2_percentile=(0.5, 99.5),  # 截断最亮的0.5%
    phase_clahe_clip=2.0,          # CLAHE对比度限制
    phase_clahe_grid=(8, 8)        # CLAHE网格大小
)

# 通道映射
pseudo_rgb = mapper.map_channels(actinin, phase, dapi)
# 输出: (H, W, 3) uint8 图像
```

**通道映射策略：**
| SAM通道 | 显微镜数据 | 作用 |
|---------|-----------|------|
| R (红) | α-actinin | 纹理特征（肌节） |
| G (绿) | Phase/明场 | 边界特征（细胞膜） |
| B (蓝) | DAPI | 定位特征（细胞核） |

### 方法二：通道适配器 (channel_adapter.py)

可学习的通道适配层，自动优化通道编码。

```python
from channel_adapter import ChannelAdapterFactory

# 三种适配器可选
adapter = ChannelAdapterFactory.create("lightweight")   # 6个参数
adapter = ChannelAdapterFactory.create("independent")   # 30个参数（推荐）
adapter = ChannelAdapterFactory.create("icvit")         # ~15000个参数

# 使用
import torch
x = torch.randn(1, 3, 1024, 1024)
y = adapter(x)
```

### 提示框生成 (prompt_generator.py)

基于SarcGraph的Z线检测 + DBSCAN聚类。

```python
from prompt_generator import SarcGraphPromptGenerator

generator = SarcGraphPromptGenerator(
    pixel_size_um=0.5,       # 像素大小
    sarcomere_length_um=2.0, # 肌节长度
    eps_factor=2.0,          # DBSCAN eps因子
    min_samples=15,          # 最小Z线数量
    padding_pixels=20        # 边界框padding
)

boxes = generator.generate_prompts(actinin_image)
```

## 🔧 训练流程

```python
from pipeline import CellSAMPipeline, CellSAMTrainer

# 1. 创建Pipeline
pipeline = CellSAMPipeline(
    sam_checkpoint="sam_vit_b.pth",
    adapter_type="independent"
)

# 2. 冻结SAM编码器（推荐）
for param in pipeline.sam.image_encoder.parameters():
    param.requires_grad = False

# 3. 创建训练器
trainer = CellSAMTrainer(
    pipeline=pipeline,
    learning_rate=1e-4,
    weight_decay=1e-5
)

# 4. 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = trainer.train_step(
            images=batch['image'],
            gt_masks=batch['mask'],
            boxes=batch['boxes']
        )
    
    # 验证
    val_loss = trainer.validate(val_loader)
    
    # 保存
    pipeline.save_adapter(f"adapter_epoch{epoch}.pth")
```

## ⚙️ 参数调优建议

### DBSCAN参数
| 参数 | 计算方法 | 说明 |
|------|---------|------|
| eps | sarcomere_length × eps_factor / pixel_size | 通常4-10像素 |
| min_samples | 10-20 | 根据细胞大小调整 |

### 数据增强建议
- 弹性变形（模拟细胞形态变异）
- 通道独立的亮度扰动
- 随机翻转和旋转

## 📊 评估指标

使用SarcGraph进行分割质量验证：

```python
# 肌节密度
density = num_zlines / mask_area

# 取向有序度 (OOP)
# 正常: OOP > 0.3
# 可疑合并: OOP < 0.15
```

## 🐛 常见问题

**Q: 检测到的细胞数为0？**
- 检查actinin通道质量（SNR > 3）
- 调整Z线检测阈值（降低threshold）
- 调整DBSCAN参数（增大eps或减小min_samples）

**Q: 边界框太大/太小？**
- 调整padding_pixels和padding_ratio
- 检查pixel_size_um是否正确

**Q: SAM分割不准确？**
- 确保使用语义通道映射
- 考虑微调通道适配器
- 检查边界框是否准确覆盖细胞

## 📚 参考文献

1. SAMCell: Generalized Label-Free Biological Cell Segmentation (PLOS One, 2025)
2. CellSAM: A Foundation Model for Cell Segmentation (Nature Methods, 2025)
3. Segment Anything for Microscopy (Nature Methods, 2024)
4. Sarc-Graph: Automated segmentation, tracking, and analysis of sarcomeres (PLOS Computational Biology, 2021)

## 📄 License

MIT License
