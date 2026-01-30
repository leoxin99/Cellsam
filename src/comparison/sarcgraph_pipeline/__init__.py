"""
CellSAM Pipeline for Cardiomyocyte Segmentation

结合方法一（数据预处理）和方法二（模型适配）的完整心肌细胞分割方案
"""

from .preprocessing import SemanticChannelMapper, QualityChecker
from .channel_adapter import (
    LightweightChannelAdapter,
    IndependentChannelAdapter,
    ICViTStyleAdapter,
    ChannelAdapterFactory
)
from .prompt_generator import (
    ZLineDetector,
    SarcGraphPromptGenerator,
    AdaptivePromptGenerator,
    BoundingBox
)
from .pipeline import (
    CellSAMPipeline,
    CellSAMTrainer,
    DiceFocalLoss,
    IoULoss,
    create_pipeline
)

__version__ = "1.0.0"
__author__ = "CellSAM Team"

__all__ = [
    # 预处理
    "SemanticChannelMapper",
    "QualityChecker",
    
    # 通道适配器
    "LightweightChannelAdapter",
    "IndependentChannelAdapter",
    "ICViTStyleAdapter",
    "ChannelAdapterFactory",
    
    # 提示生成
    "ZLineDetector",
    "SarcGraphPromptGenerator",
    "AdaptivePromptGenerator",
    "BoundingBox",
    
    # Pipeline
    "CellSAMPipeline",
    "CellSAMTrainer",
    "DiceFocalLoss",
    "IoULoss",
    "create_pipeline",
]
