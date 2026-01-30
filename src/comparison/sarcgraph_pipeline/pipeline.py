"""
完整的CellSAM Pipeline
结合方法一（数据预处理）和方法二（模型适配）

使用方法:
    from pipeline import CellSAMPipeline
    
    pipeline = CellSAMPipeline(adapter_type="independent")
    masks = pipeline.segment(actinin, phase, dapi)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

# 导入自定义模块
from preprocessing import SemanticChannelMapper, QualityChecker
from channel_adapter import ChannelAdapterFactory
from prompt_generator import SarcGraphPromptGenerator, BoundingBox


class DiceFocalLoss(nn.Module):
    """
    Dice Loss + Focal Loss 组合
    
    用于训练分割模型，特别适合处理类别不平衡问题
    """
    
    def __init__(self, 
                 dice_weight: float = 1.0,
                 focal_weight: float = 1.0,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice Loss"""
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal Loss"""
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        
        focal = alpha_t * ((1 - p_t) ** self.focal_gamma) * bce
        return focal.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算总损失
        
        参数:
            pred: 预测logits, shape (B, 1, H, W)
            target: 目标mask, shape (B, 1, H, W)
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        return self.dice_weight * dice + self.focal_weight * focal


class IoULoss(nn.Module):
    """IoU Loss"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou


class CellSAMPipeline:
    """
    完整的CellSAM Pipeline
    
    集成：
    1. 方法一：数据预处理（SemanticChannelMapper）
    2. 方法二：通道适配器（Channel Adapter）
    3. 提示生成：SarcGraph
    4. SAM分割
    """
    
    def __init__(self,
                 sam_checkpoint: Optional[str] = None,
                 sam_model_type: str = "vit_b",
                 adapter_type: str = "independent",
                 adapter_kwargs: Optional[Dict] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 pixel_size_um: float = 0.5):
        """
        参数:
            sam_checkpoint: SAM模型权重路径
            sam_model_type: SAM模型类型 ("vit_h", "vit_l", "vit_b")
            adapter_type: 通道适配器类型
            adapter_kwargs: 适配器的额外参数
            device: 计算设备
            pixel_size_um: 像素大小（微米）
        """
        self.device = device
        self.pixel_size_um = pixel_size_um
        
        # ========== 方法一：数据预处理器 ==========
        self.preprocessor = SemanticChannelMapper()
        self.quality_checker = QualityChecker()
        
        # ========== 方法二：通道适配器 ==========
        adapter_kwargs = adapter_kwargs or {}
        self.channel_adapter = ChannelAdapterFactory.create(
            adapter_type, **adapter_kwargs
        ).to(device)
        
        # ========== 提示生成器 ==========
        self.prompt_generator = SarcGraphPromptGenerator(
            pixel_size_um=pixel_size_um
        )
        
        # ========== SAM模型 ==========
        self.sam = None
        self.sam_predictor = None
        
        if sam_checkpoint is not None:
            self._load_sam(sam_checkpoint, sam_model_type)
    
    def _load_sam(self, checkpoint: str, model_type: str):
        """加载SAM模型"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
            self.sam = self.sam.to(self.device)
            self.sam_predictor = SamPredictor(self.sam)
            
            print(f"✓ SAM模型已加载: {model_type}")
        except ImportError:
            print("⚠ segment_anything 未安装，SAM功能不可用")
            print("  安装命令: pip install segment-anything")
    
    def preprocess(self,
                   actinin: np.ndarray,
                   phase: np.ndarray,
                   dapi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        完整的预处理流程
        
        返回:
            pseudo_rgb: numpy格式的伪RGB图像 (H, W, 3)
            tensor: 经过适配器处理的tensor (1, 3, H, W)
        """
        # 方法一：语义通道映射
        pseudo_rgb = self.preprocessor.map_channels(actinin, phase, dapi)
        
        # 转为tensor
        tensor = torch.from_numpy(pseudo_rgb).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor / 255.0  # 归一化到 [0, 1]
        tensor = tensor.to(self.device)
        
        # 方法二：通道适配
        with torch.no_grad():
            adapted_tensor = self.channel_adapter(tensor)
        
        return pseudo_rgb, adapted_tensor
    
    def generate_prompts(self, actinin: np.ndarray) -> List[BoundingBox]:
        """
        使用SarcGraph生成提示框
        """
        return self.prompt_generator.generate_prompts(actinin)
    
    def segment_with_boxes(self,
                           image: np.ndarray,
                           boxes: List[BoundingBox]) -> List[np.ndarray]:
        """
        使用边界框提示进行分割
        
        参数:
            image: RGB图像 (H, W, 3)
            boxes: 边界框列表
            
        返回:
            List[np.ndarray]: 每个边界框对应的掩膜
        """
        if self.sam_predictor is None:
            raise RuntimeError("SAM模型未加载，请提供checkpoint路径")
        
        # 设置图像
        self.sam_predictor.set_image(image)
        
        masks = []
        for box in boxes:
            # 转换为SAM格式
            box_array = np.array([box.to_xyxy()])
            
            # 预测
            mask, score, logit = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_array,
                multimask_output=False
            )
            
            masks.append(mask[0])
        
        return masks
    
    def segment(self,
                actinin: np.ndarray,
                phase: np.ndarray,
                dapi: Optional[np.ndarray] = None,
                return_details: bool = False) -> Dict[str, Any]:
        """
        完整的分割流程
        
        参数:
            actinin: α-actinin通道图像
            phase: 相位差/明场通道图像
            dapi: DAPI通道图像（可选）
            return_details: 是否返回详细信息
            
        返回:
            dict: 包含masks, boxes, quality_check等信息
        """
        result = {
            'masks': [],
            'boxes': [],
            'quality_check': None,
            'pseudo_rgb': None,
            'num_cells': 0
        }
        
        # 1. 质量检查
        quality = self.quality_checker.check_quality(actinin)
        result['quality_check'] = quality
        
        if not quality['passed']:
            print(f"⚠ 图像质量警告: SNR={quality['snr']:.2f}, {quality['recommendation']}")
        
        # 2. 预处理
        pseudo_rgb, adapted_tensor = self.preprocess(actinin, phase, dapi)
        result['pseudo_rgb'] = pseudo_rgb
        
        # 3. 生成提示框
        boxes = self.generate_prompts(actinin)
        result['boxes'] = boxes
        result['num_cells'] = len(boxes)
        
        if len(boxes) == 0:
            print("⚠ 未检测到有效的心肌细胞")
            return result
        
        # 4. SAM分割（如果SAM可用）
        if self.sam_predictor is not None:
            # 将adapted_tensor转回numpy用于SAM
            # 注意：SAM期望0-255的uint8图像
            adapted_np = (adapted_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            masks = self.segment_with_boxes(adapted_np, boxes)
            result['masks'] = masks
        else:
            print("⚠ SAM未加载，仅返回提示框")
        
        return result
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取所有可训练参数"""
        params = list(self.channel_adapter.parameters())
        
        # 如果SAM存在且需要训练decoder
        if self.sam is not None:
            for param in self.sam.mask_decoder.parameters():
                if param.requires_grad:
                    params.append(param)
        
        return params
    
    def train_mode(self):
        """设置为训练模式"""
        self.channel_adapter.train()
        if self.sam is not None:
            self.sam.mask_decoder.train()
    
    def eval_mode(self):
        """设置为评估模式"""
        self.channel_adapter.eval()
        if self.sam is not None:
            self.sam.eval()
    
    def save_adapter(self, path: str):
        """保存通道适配器权重"""
        torch.save(self.channel_adapter.state_dict(), path)
        print(f"✓ 适配器权重已保存到: {path}")
    
    def load_adapter(self, path: str):
        """加载通道适配器权重"""
        self.channel_adapter.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✓ 适配器权重已加载: {path}")


class CellSAMTrainer:
    """
    CellSAM训练器
    """
    
    def __init__(self,
                 pipeline: CellSAMPipeline,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        """
        参数:
            pipeline: CellSAM Pipeline实例
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.pipeline = pipeline
        self.device = pipeline.device
        
        # 损失函数
        self.criterion = DiceFocalLoss(dice_weight=1.0, focal_weight=1.0)
        
        # 优化器
        trainable_params = pipeline.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, 
                   images: torch.Tensor, 
                   gt_masks: torch.Tensor,
                   boxes: torch.Tensor) -> float:
        """
        单步训练
        
        参数:
            images: 输入图像 (B, 3, H, W)
            gt_masks: Ground Truth掩膜 (B, N, H, W)
            boxes: 边界框 (B, N, 4)
            
        返回:
            float: 损失值
        """
        self.pipeline.train_mode()
        self.optimizer.zero_grad()
        
        # 前向传播
        # 1. 通道适配
        adapted = self.pipeline.channel_adapter(images)
        
        # 2. SAM编码 (需要根据实际SAM API调整)
        # ... SAM forward pass ...
        
        # 3. 计算损失
        # loss = self.criterion(pred_masks, gt_masks)
        
        # 占位符 - 实际训练时需要完整实现
        loss = torch.tensor(0.0, requires_grad=True)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> float:
        """验证"""
        self.pipeline.eval_mode()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # ... 验证逻辑 ...
                pass
        
        return total_loss / max(count, 1)


# ============ 便捷函数 ============

def create_pipeline(adapter_type: str = "independent",
                    sam_checkpoint: Optional[str] = None,
                    pixel_size_um: float = 0.5) -> CellSAMPipeline:
    """
    创建Pipeline的便捷函数
    
    参数:
        adapter_type: "lightweight", "independent", "icvit"
        sam_checkpoint: SAM权重路径（可选）
        pixel_size_um: 像素大小
    """
    return CellSAMPipeline(
        sam_checkpoint=sam_checkpoint,
        adapter_type=adapter_type,
        pixel_size_um=pixel_size_um
    )


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 60)
    print("CellSAM Pipeline 测试")
    print("=" * 60)
    
    # 创建测试数据
    H, W = 256, 256
    actinin = np.random.randint(100, 5000, (H, W), dtype=np.uint16)
    phase = np.random.randint(50, 200, (H, W), dtype=np.uint8)
    dapi = np.random.randint(100, 3000, (H, W), dtype=np.uint16)
    
    # 添加模拟的肌节纹理
    for i in range(50, 150, 8):
        actinin[60:140, i:i+3] = 15000
    for i in range(160, 230, 8):
        actinin[100:180, i:i+3] = 15000
    
    # 创建Pipeline（不加载SAM）
    pipeline = create_pipeline(
        adapter_type="independent",
        sam_checkpoint=None,  # 不加载SAM
        pixel_size_um=0.5
    )
    
    print(f"\n设备: {pipeline.device}")
    print(f"适配器参数量: {sum(p.numel() for p in pipeline.channel_adapter.parameters())}")
    
    # 测试预处理
    print("\n--- 预处理测试 ---")
    pseudo_rgb, adapted_tensor = pipeline.preprocess(actinin, phase, dapi)
    print(f"伪RGB shape: {pseudo_rgb.shape}")
    print(f"适配后tensor shape: {adapted_tensor.shape}")
    
    # 测试提示框生成
    print("\n--- 提示框生成测试 ---")
    boxes = pipeline.generate_prompts(actinin)
    print(f"检测到 {len(boxes)} 个潜在细胞")
    for i, box in enumerate(boxes):
        print(f"  细胞{i+1}: {box.to_xyxy()}, Z线数量: {box.num_zlines}")
    
    # 测试完整流程（不含SAM分割）
    print("\n--- 完整流程测试 ---")
    result = pipeline.segment(actinin, phase, dapi)
    print(f"质量检查: {result['quality_check']}")
    print(f"检测到的细胞数: {result['num_cells']}")
    
    # 测试保存/加载
    print("\n--- 保存/加载测试 ---")
    pipeline.save_adapter("/tmp/test_adapter.pth")
    pipeline.load_adapter("/tmp/test_adapter.pth")
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
