"""
统一推理核心模块 (Unified Inference Core)

所有推理路径 (训练验证、标准评估、E2E评估) 都应调用这里的函数，
确保阈值、冲突裁决、后处理等口径一致。

Author: Claude (Antigravity) - Phase 0 Implementation
Date: 2026-02-09
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InferenceConfig:
    """统一推理配置 — 所有脚本的单一来源"""
    mask_threshold: float = 0.5
    use_sam_iou_filter: bool = False
    sam_iou_threshold: float = 0.5
    apply_box_clipping: bool = True
    box_expand: float = 0.1
    conflict_policy: str = "argmax_prob"  # argmax_prob, first_write, last_write
    apply_postprocess: bool = False
    validate_size: bool = False
    min_cell_area: int = 13884     # GT 分布 P1 @1024px (原始 40836 × 0.340)
    max_cell_area: int = 174735    # GT 分布 P99 @1024px (原始 513928 × 0.340)
    
    @classmethod
    def default(cls) -> "InferenceConfig":
        """默认推理配置 — 所有脚本都应调用此方法而非硬编码参数"""
        return cls()
    
    @classmethod
    def from_dict(cls, d: dict) -> "InferenceConfig":
        """从配置字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "InferenceConfig":
        """从 YAML 配置文件的 inference 段创建"""
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        infer_cfg = cfg.get('inference', {})
        return cls.from_dict(infer_cfg)


@dataclass
class InferenceResult:
    """推理结果"""
    instance_mask: np.ndarray  # [H, W] 实例分割结果, 每个像素是实例 ID
    confidence_map: np.ndarray  # [N, H, W] 各实例的置信度图
    n_instances: int
    conflict_pixels: int
    stats: Dict = field(default_factory=dict)


def load_cellsam_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda',
    adapter_cls=None,
    adapter_kwargs: Optional[dict] = None,
):
    """
    统一 checkpoint 加载器 (含 adapter 支持)
    
    所有评估脚本应使用此函数加载模型，确保 adapter 不被遗漏。
    
    Args:
        checkpoint_path: checkpoint 文件路径
        device: 计算设备
        adapter_cls: Adapter 类 (e.g. IndependentChannelAdapter)，None 表示不使用
        adapter_kwargs: Adapter 构造参数
    
    Returns:
        (model, adapter, checkpoint_info)
        adapter 为 None 若未提供 adapter_cls
    """
    # 延迟导入避免循环依赖
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cellSAM_source"))
    from cellSAM import get_model
    
    model = get_model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 提取模型权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # 加载 adapter
    adapter = None
    if adapter_cls is not None:
        kwargs = adapter_kwargs or {}
        adapter = adapter_cls(**kwargs)
        if isinstance(checkpoint, dict) and 'adapter_state_dict' in checkpoint:
            adapter.load_state_dict(checkpoint['adapter_state_dict'])
            print(f"  ✅ Adapter loaded from checkpoint")
        else:
            print(f"  ⚠️ No adapter_state_dict in checkpoint, using fresh adapter")
        adapter = adapter.to(device)
        adapter.eval()
    
    # 提取 checkpoint 信息
    info = {}
    if isinstance(checkpoint, dict):
        info = {
            'epoch': checkpoint.get('epoch', '?'),
            'best_dice': checkpoint.get('best_dice', 0),
            'has_adapter': 'adapter_state_dict' in checkpoint,
        }
    
    return model, adapter, info


def segment_with_boxes(
    model: torch.nn.Module,
    image: torch.Tensor,
    boxes: torch.Tensor,
    config: InferenceConfig,
    device: str = 'cuda',
    return_confidence: bool = False
) -> InferenceResult:
    """
    统一推理核心函数
    
    Args:
        model: CellSAM 模型
        image: [C, H, W] 或 [1, C, H, W] 输入图像 (已 normalize)
        boxes: [N, 4] 边界框 (x1, y1, x2, y2)
        config: 推理配置
        device: 计算设备
        return_confidence: 是否返回完整置信度图 (内存开销大)
    
    Returns:
        InferenceResult: 包含 instance_mask 和统计信息
    """
    model.eval()
    
    # 处理输入维度
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    boxes = boxes.to(device)
    
    B, C, H, W = image.shape
    N = boxes.shape[0]
    
    # 初始化输出
    all_masks = []  # 存储各实例的 sigmoid 预测
    all_boxes = []
    
    with torch.no_grad():
        # SAM 预处理和图像编码
        img_preprocessed = model.sam_preprocess(image)
        image_embedding = model.model.image_encoder(img_preprocessed)
        
        for i in range(N):
            box = boxes[i:i+1].unsqueeze(0)  # [1, 1, 4]
            
            # Prompt encoding
            sparse_emb, dense_emb = model.model.prompt_encoder(
                points=None, boxes=box, masks=None
            )
            
            # Mask decoding
            low_res_masks, iou_pred = model.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            
            # 上采样到原图尺寸
            upscaled = F.interpolate(
                low_res_masks,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )
            pred_sigmoid = torch.sigmoid(upscaled[0, 0])  # [H, W]
            
            # SAM iou 过滤
            if config.use_sam_iou_filter and iou_pred.item() < config.sam_iou_threshold:
                continue
            
            # Box clipping (with box_expand margin)
            if config.apply_box_clipping:
                x1, y1, x2, y2 = [int(c.item()) for c in boxes[i]]
                bw, bh = x2 - x1, y2 - y1
                expand = config.box_expand
                x1_clip = max(0, int(x1 - bw * expand))
                y1_clip = max(0, int(y1 - bh * expand))
                x2_clip = min(W, int(x2 + bw * expand))
                y2_clip = min(H, int(y2 + bh * expand))
                
                mask_clipped = torch.zeros_like(pred_sigmoid)
                mask_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = pred_sigmoid[y1_clip:y2_clip, x1_clip:x2_clip]
                pred_sigmoid = mask_clipped
            
            all_masks.append(pred_sigmoid.cpu())
            all_boxes.append(boxes[i].cpu())
    
    # 冲突裁决 + 实例分配
    n_valid = len(all_masks)
    if n_valid == 0:
        return InferenceResult(
            instance_mask=np.zeros((H, W), dtype=np.int32),
            confidence_map=np.zeros((0, H, W), dtype=np.float32),
            n_instances=0,
            conflict_pixels=0
        )
    
    # 堆叠所有预测
    stacked = torch.stack(all_masks, dim=0)  # [N, H, W]
    
    # 冲突裁决
    instance_mask, conflict_pixels = resolve_conflicts(
        stacked.numpy(),
        config.mask_threshold,
        config.conflict_policy
    )
    
    # 后处理 (可选)
    if config.apply_postprocess:
        instance_mask = postprocess_instance_mask(instance_mask, config)
    
    # 计算统计
    n_instances = int(instance_mask.max())
    
    result = InferenceResult(
        instance_mask=instance_mask,
        confidence_map=stacked.numpy() if return_confidence else np.zeros((0, H, W)),
        n_instances=n_instances,
        conflict_pixels=conflict_pixels
    )
    
    return result


def resolve_conflicts(
    pred_stack: np.ndarray,
    threshold: float,
    policy: str
) -> Tuple[np.ndarray, int]:
    """
    解决多实例冲突像素归属
    
    Args:
        pred_stack: [N, H, W] 各实例的 sigmoid 预测
        threshold: 二值化阈值
        policy: 冲突裁决策略
    
    Returns:
        instance_mask: [H, W] 实例分割结果
        conflict_pixels: 冲突像素数量
    """
    N, H, W = pred_stack.shape
    
    # 二值化
    binary_stack = (pred_stack > threshold).astype(np.int32)
    
    # 计算冲突像素
    overlap_count = binary_stack.sum(axis=0)  # [H, W]
    conflict_pixels = int((overlap_count >= 2).sum())
    
    if policy == "argmax_prob":
        # 取置信度最高的实例
        max_prob_idx = np.argmax(pred_stack, axis=0)  # [H, W]
        # 只保留超过阈值的
        any_above_thresh = (overlap_count >= 1)
        instance_mask = np.where(any_above_thresh, max_prob_idx + 1, 0).astype(np.int32)
        
    elif policy == "first_write":
        # 先处理的 box 优先
        instance_mask = np.zeros((H, W), dtype=np.int32)
        for i in range(N):
            mask = binary_stack[i] > 0
            new_pixels = mask & (instance_mask == 0)
            instance_mask[new_pixels] = i + 1
            
    elif policy == "last_write":
        # 后处理的 box 覆盖
        instance_mask = np.zeros((H, W), dtype=np.int32)
        for i in range(N):
            mask = binary_stack[i] > 0
            instance_mask[mask] = i + 1
            
    else:
        raise ValueError(f"Unknown conflict policy: {policy}")
    
    return instance_mask, conflict_pixels


def postprocess_instance_mask(
    instance_mask: np.ndarray,
    config: InferenceConfig
) -> np.ndarray:
    """
    实例 mask 后处理
    
    Args:
        instance_mask: [H, W] 实例分割结果
        config: 推理配置
    
    Returns:
        处理后的 instance_mask
    """
    from scipy import ndimage
    
    result = np.zeros_like(instance_mask)
    new_id = 0
    
    for cell_id in range(1, instance_mask.max() + 1):
        cell_mask = (instance_mask == cell_id)
        
        # 连通组件分析 - 保留最大组件
        labeled, n_components = ndimage.label(cell_mask)
        if n_components == 0:
            continue
        
        component_sizes = ndimage.sum(cell_mask, labeled, range(1, n_components + 1))
        largest_idx = np.argmax(component_sizes) + 1
        largest_mask = (labeled == largest_idx)
        
        # 尺寸验证
        if config.validate_size:
            area = largest_mask.sum()
            if area < config.min_cell_area or area > config.max_cell_area:
                continue
        
        new_id += 1
        result[largest_mask] = new_id
    
    return result


# ============== 便捷函数 ==============

def get_default_config() -> InferenceConfig:
    """获取默认推理配置"""
    return InferenceConfig()


def compute_intrusion_rate(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    cell_id: int
) -> float:
    """
    计算单个细胞的侵占率
    
    Args:
        pred_mask: [H, W] 预测 mask (二值或实例)
        gt_mask: [H, W] GT 实例 mask
        cell_id: 当前细胞 ID
    
    Returns:
        intrusion_rate: 侵占邻居区域像素 / 总前景像素
    """
    if pred_mask.ndim == 2 and pred_mask.dtype == bool:
        pred_fg = pred_mask
    else:
        pred_fg = pred_mask > 0.5
    
    # 邻居区域 = 其他细胞的 GT
    neighbor_region = (gt_mask > 0) & (gt_mask != cell_id)
    
    intrusion = (pred_fg & neighbor_region).sum()
    total_fg = pred_fg.sum()
    
    if total_fg == 0:
        return 0.0
    
    return float(intrusion / total_fg)


def compute_conflict_rate(pred_stack: np.ndarray, threshold: float = 0.5) -> float:
    """
    计算重叠冲突率
    
    Args:
        pred_stack: [N, H, W] 各实例的 sigmoid 预测
        threshold: 二值化阈值
    
    Returns:
        conflict_rate: 重叠像素数 / 总前景像素数
    """
    binary_stack = (pred_stack > threshold).astype(np.int32)
    overlap_count = binary_stack.sum(axis=0)
    
    conflict_pixels = (overlap_count >= 2).sum()
    total_fg_pixels = (overlap_count >= 1).sum()
    
    if total_fg_pixels == 0:
        return 0.0
    
    return float(conflict_pixels / total_fg_pixels)
