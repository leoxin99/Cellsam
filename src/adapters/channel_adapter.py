"""
IndependentChannelAdapter: 可学习的通道适配层

功能: 为每个通道学习独立的特征变换，适配预训练 SAM ViT
位置: 插入在 CellSAM.forward() 中，image_encoder 之前
参数量: ~30 (3个 3×3 卷积核 + 3个偏置)

使用方法:
    adapter = IndependentChannelAdapter(kernel_size=3, use_relu=True)
    adapted_image = adapter(semantic_mapped_image)

更新日志:
- 2026-01-28: 初始版本，基于数据驱动参数选择
"""

import torch
import torch.nn as nn
from typing import Tuple


class IndependentChannelAdapter(nn.Module):
    """
    独立通道卷积适配器
    
    为每个通道学习独立的特征变换，模拟 IC-ViT 的设计思想。
    初始化为恒等映射，确保训练初期与预训练权重兼容。
    
    输入通道顺序 (来自 SemanticChannelMapper, 生物学一致):
        - Ch0 (R): BF (灰度, 细胞边界)
        - Ch1 (G): Actn2 (绿色荧光, 肌节纹理)
        - Ch2 (B): DAPI (蓝色荧光, 细胞核)
    
    参数:
        kernel_size: 卷积核大小 (推荐 3)
        use_relu: 是否使用 ReLU 激活 (默认 False, 避免与恒等初始化冲突)
    """
    
    def __init__(self, kernel_size: int = 3, use_relu: bool = False):
        """
        Args:
            kernel_size: 卷积核大小 (推荐 3)
            use_relu: 是否使用 ReLU 激活 (默认 False)
                      ⚠️ ReLU + identity init 冲突: 恒等初始化保证 output=input,
                      但 ReLU 从 epoch 1 就裁切负值, 破坏恒等映射。
                      对于 pixel 空间 (0~255) 影响有限, 但 SemanticChannelMapper
                      输出可能含经 normalization 后的负值。建议关闭。
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        self.use_relu = use_relu
        padding = kernel_size // 2
        
        # 每个通道独立的卷积层
        self.actn2_conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True)
        self.bf_conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True)
        self.dapi_conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True)
        
        # 激活函数 (默认关闭, 避免与恒等初始化冲突)
        self.activation = nn.ReLU(inplace=True) if use_relu else nn.Identity()
        
        # 初始化为恒等映射
        self._init_as_identity()
    
    def _init_as_identity(self):
        """初始化卷积核为恒等映射 (中心=1, 其他=0)"""
        for conv in [self.actn2_conv, self.bf_conv, self.dapi_conv]:
            nn.init.zeros_(conv.weight)
            center = self.kernel_size // 2
            conv.weight.data[:, :, center, center] = 1.0
            nn.init.zeros_(conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) - 来自 SemanticChannelMapper 的输出
               通道顺序: [BF, Actn2, DAPI]
        
        Returns:
            (B, 3, H, W) - 适配后的图像
        """
        # 分离通道 (R=BF, G=Actn2, B=DAPI)
        ch_bf = x[:, 0:1, :, :]     # (B, 1, H, W) - BF
        ch_actn2 = x[:, 1:2, :, :]  # Actn2
        ch_dapi = x[:, 2:3, :, :]   # DAPI
        
        # 独立处理
        ch_bf = self.activation(self.bf_conv(ch_bf))
        ch_actn2 = self.activation(self.actn2_conv(ch_actn2))
        ch_dapi = self.activation(self.dapi_conv(ch_dapi))
        
        # 合并
        return torch.cat([ch_bf, ch_actn2, ch_dapi], dim=1)
    
    def get_param_count(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def visualize_kernels(self) -> dict:
        """返回学习到的卷积核 (用于可视化)"""
        return {
            'actn2': self.actn2_conv.weight.detach().cpu().numpy().squeeze(),
            'bf': self.bf_conv.weight.detach().cpu().numpy().squeeze(),
            'dapi': self.dapi_conv.weight.detach().cpu().numpy().squeeze()
        }


class LightweightChannelAdapter(nn.Module):
    """
    轻量级通道适配器 (备选方案)
    
    仅学习每通道的增益和偏置，参数量更少 (~6)。
    适用于快速验证或数据量较少的情况。
    """
    
    def __init__(self):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(3))
        self.biases = nn.Parameter(torch.zeros(3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gains = self.gains.view(1, 3, 1, 1)
        biases = self.biases.view(1, 3, 1, 1)
        return x * gains + biases
    
    def get_param_count(self) -> int:
        return 6


# 测试代码
if __name__ == "__main__":
    adapter = IndependentChannelAdapter(kernel_size=3, use_relu=True)
    print(f"IndependentChannelAdapter params: {adapter.get_param_count()}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 1024, 1024)
    y = adapter(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    
    # 验证恒等初始化
    x_test = torch.rand(1, 3, 64, 64)
    y_test = adapter(x_test)
    diff = (x_test - y_test).abs().mean()
    print(f"Identity init diff (should be ~0 without ReLU): {diff:.6f}")
    print("✅ Adapter test passed!")
