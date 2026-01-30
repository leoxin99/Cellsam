"""
方法二：模型架构层面的通道适配器
在SAM的Image Encoder之前插入可学习的适配层

使用方法:
    from channel_adapter import LightweightChannelAdapter, IndependentChannelAdapter
    
    adapter = IndependentChannelAdapter(kernel_size=3)
    adapted_image = adapter(image_tensor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LightweightChannelAdapter(nn.Module):
    """
    轻量级通道适配器
    
    参数量：仅 6 个可学习参数（每个通道一个增益+偏置）
    作用：学习每个通道的最优权重
    
    适用场景：快速验证、小数据集
    """
    
    def __init__(self):
        super().__init__()
        # 每个通道的可学习增益（初始化为1）
        self.channel_gains = nn.Parameter(torch.ones(3))
        # 每个通道的可学习偏置（初始化为0）
        self.channel_biases = nn.Parameter(torch.zeros(3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x, shape (B, 3, H, W)
        输出: shape (B, 3, H, W)
        """
        # 对每个通道应用增益和偏置
        # gains: (3,) -> (1, 3, 1, 1) 用于广播
        gains = self.channel_gains.view(1, 3, 1, 1)
        biases = self.channel_biases.view(1, 3, 1, 1)
        
        return x * gains + biases
    
    def get_channel_weights(self) -> dict:
        """获取当前学习到的通道权重"""
        return {
            'gains': self.channel_gains.detach().cpu().numpy(),
            'biases': self.channel_biases.detach().cpu().numpy()
        }


class IndependentChannelAdapter(nn.Module):
    """
    独立通道卷积适配器（模拟IC-ViT的思想）
    
    参数量：约 3 × (k×k + 1) 个参数
    作用：为每个通道学习独立的特征变换
    
    适用场景：中等数据量（100-500张图像）
    """
    
    def __init__(self, kernel_size: int = 3, use_relu: bool = True):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.use_relu = use_relu
        padding = kernel_size // 2
        
        # 为每个通道创建独立的卷积层
        # Actinin通道：学习纹理增强
        self.actn2_conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True)
        
        # Phase通道：学习边缘增强
        self.phase_conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True)
        
        # DAPI通道：学习斑点增强
        self.dapi_conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True)
        
        # 可选的激活函数
        self.activation = nn.ReLU(inplace=True) if use_relu else nn.Identity()
        
        # 初始化为恒等映射
        self._init_as_identity()
    
    def _init_as_identity(self):
        """初始化卷积核为近似恒等映射"""
        for conv in [self.actn2_conv, self.phase_conv, self.dapi_conv]:
            nn.init.zeros_(conv.weight)
            # 中心位置设为1
            center = self.kernel_size // 2
            conv.weight.data[:, :, center, center] = 1.0
            nn.init.zeros_(conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x, shape (B, 3, H, W)
              通道顺序: [Actinin, Phase, DAPI]
        输出: shape (B, 3, H, W)
        """
        # 分离通道
        ch_actn2 = x[:, 0:1, :, :]  # (B, 1, H, W)
        ch_phase = x[:, 1:2, :, :]
        ch_dapi = x[:, 2:3, :, :]
        
        # 独立处理
        ch_actn2 = self.activation(self.actn2_conv(ch_actn2))
        ch_phase = self.activation(self.phase_conv(ch_phase))
        ch_dapi = self.activation(self.dapi_conv(ch_dapi))
        
        # 合并
        return torch.cat([ch_actn2, ch_phase, ch_dapi], dim=1)
    
    def visualize_kernels(self) -> dict:
        """可视化学习到的卷积核"""
        return {
            'actn2_kernel': self.actn2_conv.weight.detach().cpu().numpy().squeeze(),
            'phase_kernel': self.phase_conv.weight.detach().cpu().numpy().squeeze(),
            'dapi_kernel': self.dapi_conv.weight.detach().cpu().numpy().squeeze()
        }


class ICViTStyleAdapter(nn.Module):
    """
    IC-ViT风格的通道适配器（最接近CellSAM的实现）
    
    包含：
    1. 独立通道的特征提取
    2. 通道混合注意力
    3. 特征融合
    
    参数量：较大（~15000），建议配合LoRA使用
    适用场景：大数据量（>500张图像）
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # ========== 1. 独立通道编码器 ==========
        self.channel_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, hidden_dim // 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(in_channels)
        ])
        
        # ========== 2. 通道混合注意力 ==========
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # ========== 3. 输出投影（回到3通道） ==========
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim * in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1)
        )
        
        # ========== 4. 残差连接权重 ==========
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x, shape (B, 3, H, W)
        输出: shape (B, 3, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. 独立通道编码
        channel_features = []
        for i in range(C):
            ch_feat = self.channel_encoders[i](x[:, i:i+1, :, :])  # (B, hidden_dim, H, W)
            channel_features.append(ch_feat)
        
        # 2. 准备注意力计算
        # 将特征reshape为 (B*H*W, C, hidden_dim)
        feat_list = []
        for feat in channel_features:
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)
            feat_list.append(feat_flat)
        
        # Stack along channel dimension: (B, H*W, C, hidden_dim)
        feat_stack = torch.stack(feat_list, dim=2)
        
        # Reshape for attention: (B*H*W, C, hidden_dim)
        feat_for_attn = feat_stack.reshape(B * H * W, C, self.hidden_dim)
        
        # 3. 通道混合注意力
        attn_out, _ = self.channel_attention(
            feat_for_attn, feat_for_attn, feat_for_attn
        )
        attn_out = self.attn_norm(attn_out + feat_for_attn)  # 残差连接
        
        # 4. Reshape回图像格式
        # (B*H*W, C, hidden_dim) -> (B, H*W, C, hidden_dim) -> (B, C*hidden_dim, H, W)
        attn_out = attn_out.reshape(B, H * W, C, self.hidden_dim)
        attn_out = attn_out.permute(0, 2, 3, 1).reshape(B, C * self.hidden_dim, H, W)
        
        # 5. 输出投影
        out = self.output_proj(attn_out)  # (B, 3, H, W)
        
        # 6. 残差连接
        out = x + self.residual_weight * out
        
        return out


class ChannelAdapterFactory:
    """
    通道适配器工厂类
    根据配置创建适当的适配器
    """
    
    @staticmethod
    def create(adapter_type: str, **kwargs) -> nn.Module:
        """
        创建通道适配器
        
        参数:
            adapter_type: "lightweight", "independent", "icvit", "none"
            **kwargs: 传递给具体适配器的参数
            
        返回:
            nn.Module: 通道适配器实例
        """
        if adapter_type == "lightweight":
            return LightweightChannelAdapter()
        
        elif adapter_type == "independent":
            kernel_size = kwargs.get('kernel_size', 3)
            use_relu = kwargs.get('use_relu', True)
            return IndependentChannelAdapter(kernel_size=kernel_size, use_relu=use_relu)
        
        elif adapter_type == "icvit":
            hidden_dim = kwargs.get('hidden_dim', 64)
            num_heads = kwargs.get('num_heads', 4)
            dropout = kwargs.get('dropout', 0.1)
            return ICViTStyleAdapter(
                hidden_dim=hidden_dim, 
                num_heads=num_heads,
                dropout=dropout
            )
        
        elif adapter_type == "none":
            return nn.Identity()
        
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}. "
                           f"Choose from: lightweight, independent, icvit, none")
    
    @staticmethod
    def get_param_count(adapter: nn.Module) -> int:
        """获取适配器的参数数量"""
        return sum(p.numel() for p in adapter.parameters() if p.requires_grad)


# ============ 测试代码 ============
if __name__ == "__main__":
    # 测试所有适配器
    x = torch.randn(2, 3, 256, 256)
    
    print("=" * 60)
    print("通道适配器测试")
    print("=" * 60)
    
    for adapter_type in ["lightweight", "independent", "icvit"]:
        print(f"\n--- {adapter_type} ---")
        
        adapter = ChannelAdapterFactory.create(adapter_type, hidden_dim=32, num_heads=4)
        
        # 前向传播
        y = adapter(x)
        
        # 统计
        param_count = ChannelAdapterFactory.get_param_count(adapter)
        
        print(f"输入 shape: {x.shape}")
        print(f"输出 shape: {y.shape}")
        print(f"参数量: {param_count}")
        
        # 测试梯度
        loss = y.mean()
        loss.backward()
        print(f"梯度测试: OK")
        
        # 清理梯度
        adapter.zero_grad()
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
