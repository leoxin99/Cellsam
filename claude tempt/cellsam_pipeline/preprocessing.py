"""
方法一：数据层面的语义通道映射
将显微镜多通道图像转换为SAM友好的伪RGB格式

使用方法:
    from preprocessing import SemanticChannelMapper
    
    mapper = SemanticChannelMapper()
    pseudo_rgb = mapper.map_channels(actinin, phase, dapi)
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class SemanticChannelMapper:
    """
    数据层面的语义通道映射器
    将显微镜多通道图像转换为SAM友好的伪RGB格式
    
    通道映射策略：
        R (通道0) ← Actinin：SAM对红色通道敏感，放置最重要的纹理信息
        G (通道1) ← Phase：绿色通道用于边界检测
        B (通道2) ← DAPI：蓝色通道用于实例分离
    """
    
    def __init__(self, 
                 actn2_percentile: Tuple[float, float] = (0.5, 99.5),
                 phase_clahe_clip: float = 2.0,
                 phase_clahe_grid: Tuple[int, int] = (8, 8),
                 output_range: Tuple[int, int] = (0, 255)):
        """
        参数:
            actn2_percentile: Actinin通道的百分位截断范围
            phase_clahe_clip: 相位差CLAHE的对比度限制
            phase_clahe_grid: CLAHE的网格大小
            output_range: 输出像素值范围
        """
        self.actn2_percentile = actn2_percentile
        self.phase_clahe_clip = phase_clahe_clip
        self.phase_clahe_grid = phase_clahe_grid
        self.output_range = output_range
        
        # 创建CLAHE对象
        self.clahe = cv2.createCLAHE(
            clipLimit=phase_clahe_clip, 
            tileGridSize=phase_clahe_grid
        )
    
    def process_actinin(self, img: np.ndarray) -> np.ndarray:
        """
        处理α-actinin通道（肌节纹理）
        
        策略：百分位截断 + 线性归一化
        目的：突出肌节条纹，抑制噪声
        
        参数:
            img: 输入图像，可以是8-bit或16-bit
            
        返回:
            np.ndarray: 处理后的8-bit图像
        """
        img = img.astype(np.float32)
        
        # 计算百分位数
        p_low = np.percentile(img, self.actn2_percentile[0])
        p_high = np.percentile(img, self.actn2_percentile[1])
        
        # 截断
        img_clipped = np.clip(img, p_low, p_high)
        
        # 归一化到输出范围
        img_norm = (img_clipped - p_low) / (p_high - p_low + 1e-8)
        img_out = (img_norm * self.output_range[1]).astype(np.uint8)
        
        return img_out
    
    def process_phase(self, img: np.ndarray) -> np.ndarray:
        """
        处理相位差/明场通道（细胞边界）
        
        策略：CLAHE自适应直方图均衡化
        目的：增强微弱的细胞膜边缘
        
        参数:
            img: 输入图像
            
        返回:
            np.ndarray: CLAHE增强后的8-bit图像
        """
        # 确保是8位图像
        if img.dtype != np.uint8:
            img_min, img_max = img.min(), img.max()
            img = ((img - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
        
        # 应用CLAHE
        img_enhanced = self.clahe.apply(img)
        
        return img_enhanced
    
    def process_dapi(self, img: np.ndarray, smooth: bool = True) -> np.ndarray:
        """
        处理DAPI通道（细胞核）
        
        策略：简单归一化 + 可选的高斯平滑
        目的：提供细胞中心的定位信息
        
        参数:
            img: 输入图像
            smooth: 是否应用高斯平滑
            
        返回:
            np.ndarray: 处理后的8-bit图像
        """
        img = img.astype(np.float32)
        
        # 归一化
        img_min, img_max = img.min(), img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)
        img_out = (img_norm * self.output_range[1]).astype(np.uint8)
        
        # 轻微平滑以减少噪声
        if smooth:
            img_out = cv2.GaussianBlur(img_out, (3, 3), 0)
        
        return img_out
    
    def map_channels(self, 
                     actinin: np.ndarray, 
                     phase: np.ndarray, 
                     dapi: Optional[np.ndarray] = None) -> np.ndarray:
        """
        将三个通道映射为伪RGB图像
        
        参数:
            actinin: α-actinin通道图像
            phase: 相位差/明场通道图像
            dapi: DAPI通道图像（可选）
        
        返回:
            np.ndarray: shape (H, W, 3), dtype uint8
        """
        # 处理各通道
        ch_r = self.process_actinin(actinin)
        ch_g = self.process_phase(phase)
        
        if dapi is not None:
            ch_b = self.process_dapi(dapi)
        else:
            # 如果没有DAPI，用零填充
            ch_b = np.zeros_like(ch_r)
        
        # 堆叠为RGB格式
        rgb_image = np.stack([ch_r, ch_g, ch_b], axis=-1)
        
        return rgb_image
    
    def apply_sam_normalization(self, img: np.ndarray) -> np.ndarray:
        """
        应用SAM标准的ImageNet归一化
        
        注意：这一步通常在SAM内部自动完成，
              如果使用SAM的标准预处理器则不需要手动调用
              
        参数:
            img: shape (H, W, 3), dtype uint8
            
        返回:
            np.ndarray: 归一化后的float32图像
        """
        # SAM/ImageNet 标准化参数
        pixel_mean = np.array([123.675, 116.28, 103.53])
        pixel_std = np.array([58.395, 57.12, 57.375])
        
        # 转为float并归一化
        img_float = img.astype(np.float32)
        img_normalized = (img_float - pixel_mean) / pixel_std
        
        return img_normalized


class QualityChecker:
    """
    图像质量检查器
    在预处理前评估图像质量，决定是否需要备用策略
    """
    
    def __init__(self, snr_threshold: float = 3.0):
        """
        参数:
            snr_threshold: 信噪比阈值，低于此值认为质量差
        """
        self.snr_threshold = snr_threshold
    
    def compute_snr(self, img: np.ndarray) -> float:
        """
        计算图像的信噪比 (Signal-to-Noise Ratio)
        
        使用方法：SNR = mean(signal) / std(background)
        """
        img = img.astype(np.float32)
        
        # 简单的前景/背景分离（Otsu阈值）
        if img.dtype != np.uint8:
            img_8bit = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        else:
            img_8bit = img
        
        threshold, _ = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 计算前景和背景
        foreground = img[img_8bit > threshold]
        background = img[img_8bit <= threshold]
        
        if len(background) == 0 or np.std(background) == 0:
            return float('inf')
        
        snr = np.mean(foreground) / (np.std(background) + 1e-8)
        return snr
    
    def check_quality(self, actinin: np.ndarray) -> dict:
        """
        检查actinin通道的质量
        
        返回:
            dict: 包含质量指标和是否通过的判断
        """
        snr = self.compute_snr(actinin)
        
        return {
            'snr': snr,
            'passed': snr >= self.snr_threshold,
            'recommendation': 'OK' if snr >= self.snr_threshold else 'Consider backup strategy'
        }


# ============ 测试代码 ============
if __name__ == "__main__":
    # 模拟测试数据
    H, W = 512, 512
    
    # 模拟16-bit荧光图像
    actinin = np.random.randint(100, 5000, (H, W), dtype=np.uint16)
    # 添加一些"肌节"纹理
    for i in range(0, H, 20):
        actinin[i:i+5, :] += 10000
    
    phase = np.random.randint(50, 200, (H, W), dtype=np.uint8)
    dapi = np.random.randint(100, 3000, (H, W), dtype=np.uint16)
    
    # 创建映射器
    mapper = SemanticChannelMapper(
        actn2_percentile=(0.5, 99.5),
        phase_clahe_clip=2.0,
        phase_clahe_grid=(8, 8)
    )
    
    # 执行映射
    pseudo_rgb = mapper.map_channels(actinin, phase, dapi)
    
    print(f"输入 actinin shape: {actinin.shape}, dtype: {actinin.dtype}")
    print(f"输入 phase shape: {phase.shape}, dtype: {phase.dtype}")
    print(f"输入 dapi shape: {dapi.shape}, dtype: {dapi.dtype}")
    print(f"输出 pseudo_rgb shape: {pseudo_rgb.shape}, dtype: {pseudo_rgb.dtype}")
    print(f"输出像素范围: [{pseudo_rgb.min()}, {pseudo_rgb.max()}]")
    
    # 质量检查
    checker = QualityChecker(snr_threshold=3.0)
    quality = checker.check_quality(actinin)
    print(f"质量检查: {quality}")
    
    # 保存测试图像
    cv2.imwrite('/tmp/test_pseudo_rgb.png', cv2.cvtColor(pseudo_rgb, cv2.COLOR_RGB2BGR))
    print("测试图像已保存到 /tmp/test_pseudo_rgb.png")
