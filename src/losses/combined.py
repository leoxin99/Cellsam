"""
Loss functions for CellSAM training.
Includes DiceLoss, BoundaryLoss, and CombinedLoss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation."""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        """Dice loss with optional mask for region-specific computation."""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        pred = pred.contiguous().reshape(-1)
        target = target.contiguous().reshape(-1).float()
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class BoundaryLoss(nn.Module):
    """
    Boundary Loss: focuses training on pixels near cell edges.
    
    This addresses the issue of high Dice but low instance IoU by emphasizing
    correct boundary prediction.
    
    Reference: Kervadec et al., "Boundary loss for highly unbalanced segmentation"
    """
    
    def __init__(self, boundary_width=3):
        super().__init__()
        self.boundary_width = boundary_width
    
    def get_boundary_mask(self, mask):
        """Extract boundary pixels from a mask using morphological operations."""
        from scipy import ndimage
        from skimage import morphology
        
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask
        
        mask_binary = (mask_np > 0.5).astype(np.float32)
        struct = morphology.disk(self.boundary_width)
        eroded = ndimage.binary_erosion(mask_binary, struct).astype(np.float32)
        boundary = mask_binary - eroded
        
        return boundary
    
    def forward(self, pred, target):
        """
        Compute boundary-focused loss.
        
        Args:
            pred: (H, W) or (B, H, W) prediction probabilities (after sigmoid)
            target: (H, W) or (B, H, W) binary ground truth
        """
        if target.dim() == 2:
            boundary_mask = self.get_boundary_mask(target)
        else:
            boundaries = [self.get_boundary_mask(target[i]) for i in range(target.shape[0])]
            boundary_mask = np.stack(boundaries)
        
        boundary_tensor = torch.from_numpy(boundary_mask).to(pred.device).float()
        n_boundary = boundary_tensor.sum()
        
        if n_boundary > 0:
            boundary_pred = pred[boundary_tensor > 0]
            boundary_target = target[boundary_tensor > 0].float()
            
            boundary_bce = F.binary_cross_entropy(
                boundary_pred.reshape(-1),
                boundary_target.reshape(-1),
                reduction='mean'
            )
            
            intersection = (boundary_pred * boundary_target).sum()
            boundary_dice = 1 - (2. * intersection + 1) / (boundary_pred.sum() + boundary_target.sum() + 1)
            
            return 0.5 * boundary_bce + 0.5 * boundary_dice
        else:
            return torch.tensor(0.0, device=pred.device)


class AJILoss(nn.Module):
    """
    Aggregated Jaccard Index (AJI) inspired loss.
    
    AJI penalizes both over-segmentation and under-segmentation at the instance level.
    For training, we approximate AJI by computing IoU between predicted and target
    regions within the bounding box context.
    
    AJI = sum(|C_i ∩ S_σ(i)|) / sum(|C_i ∪ S_σ(i)| + |S_k - matched|)
    
    Where:
    - C_i: ground truth cell i
    - S_σ(i): best matching predicted cell
    - S_k - matched: unmatched predicted cells (false positives)
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target, pred_instances=None, gt_instances=None):
        """
        Compute AJI-inspired loss.
        
        For single-cell training (one box at a time), this approximates to an IoU-like loss
        that penalizes predictions that don't match the target well.
        
        Args:
            pred: (H, W) prediction probabilities (after sigmoid), values [0, 1]
            target: (H, W) binary ground truth for the current cell
            pred_instances: Optional instance mask for multi-cell AJI
            gt_instances: Optional GT instance mask for multi-cell AJI
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1).float()
        
        # Compute soft IoU (Jaccard) loss
        # IoU = intersection / union
        intersection = (pred * target).sum()
        pred_sum = pred.sum()
        target_sum = target.sum()
        union = pred_sum + target_sum - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Additional penalty for false positives (over-segmentation)
        # Pixels predicted as foreground but not in target
        false_positive_penalty = pred * (1 - target)
        fp_weight = false_positive_penalty.sum() / (pred_sum + self.smooth)
        
        # Additional penalty for false negatives (under-segmentation)
        # Pixels in target but not predicted
        false_negative_penalty = (1 - pred) * target
        fn_weight = false_negative_penalty.sum() / (target_sum + self.smooth)
        
        # AJI loss = 1 - IoU + penalties for over/under segmentation
        # The penalties make this more sensitive to instance-level errors
        aji_loss = 1 - iou + 0.1 * (fp_weight + fn_weight)
        
        return aji_loss


class TopologyLoss(nn.Module):
    """
    Topology constraint loss for cell segmentation.
    
    Penalizes:
    1. Small fragments (connected components smaller than min_size)
    2. Multiple disconnected regions per prediction
    
    Based on E17 analysis: min_size=40836 (P1)
    """
    
    def __init__(self, min_size: int = 40836, max_components: int = 1):
        """
        Args:
            min_size: Minimum valid component size (from E17 P1)
            max_components: Expected number of components per cell (usually 1)
        """
        super().__init__()
        self.min_size = min_size
        self.max_components = max_components
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """
        Compute topology loss.
        
        Args:
            pred: Prediction probabilities (after sigmoid), shape (H, W)
            target: Not used, for API consistency
        
        Returns:
            Topology loss value
        """
        from scipy import ndimage
        
        # Binarize prediction
        pred_binary = (pred > 0.5).float().cpu().numpy()
        
        # Label connected components
        labeled, n_components = ndimage.label(pred_binary)
        
        if n_components == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Fragment penalty: count small components
        fragment_count = 0
        total_fragment_area = 0
        
        for i in range(1, n_components + 1):
            component_size = (labeled == i).sum()
            if component_size < self.min_size:
                fragment_count += 1
                total_fragment_area += component_size
        
        # Penalty based on fragment ratio
        fragment_penalty = fragment_count / max(n_components, 1)
        
        # Component count penalty: penalize multiple disconnected regions
        component_penalty = max(0, n_components - self.max_components) / max(n_components, 1)
        
        # Combined penalty
        loss = 0.5 * fragment_penalty + 0.5 * component_penalty
        
        return torch.tensor(loss, device=pred.device, dtype=pred.dtype)


class SizeLoss(nn.Module):
    """
    Size constraint loss for cell segmentation.
    
    Penalizes predictions that deviate from target cell size,
    encouraging the model to learn correct cell boundaries.
    
    Based on GT analysis (FULL dataset: 478 images, 5173 cells):
    - P1: 40836, P99: 513928 (excludes annotation errors)
    - Median: 142316 pixels
    """
    
    def __init__(self, min_area: int = 40836, max_area: int = 513928, 
                 smooth: float = 1.0, margin: float = 0.2):
        """
        Args:
            min_area: Minimum expected cell area (P1 from E17)
            max_area: Maximum expected cell area (P99 from E17)
            smooth: Smoothing factor for area ratio
            margin: Soft margin percentage (0.2 = 20% transition zone)
        """
        super().__init__()
        self.min_area = min_area
        self.max_area = max_area
        self.smooth = smooth
        self.margin = margin
        
        # Soft boundaries
        self.soft_min = int(min_area * (1 - margin))  # 20% below P1
        self.soft_max = int(max_area * (1 + margin))  # 20% above P99
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute size loss with soft thresholds.
        
        Penalty is:
        - 0 if area within [min_area, max_area]
        - Linear 0->1 in transition zones [soft_min, min_area] and [max_area, soft_max]
        - 1 if outside soft boundaries
        """
        pred_area = pred.sum()
        target_area = target.sum()
        
        # Relative size difference (always apply)
        size_diff = torch.abs(pred_area - target_area) / (target_area + self.smooth)
        
        # Soft boundary penalty
        boundary_penalty = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        if pred_area < self.soft_min:
            # Fully outside lower bound
            boundary_penalty = torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
        elif pred_area < self.min_area:
            # In lower transition zone: linear interpolation
            boundary_penalty = (self.min_area - pred_area) / (self.min_area - self.soft_min)
        elif pred_area > self.soft_max:
            # Fully outside upper bound
            boundary_penalty = torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
        elif pred_area > self.max_area:
            # In upper transition zone: linear interpolation
            boundary_penalty = (pred_area - self.max_area) / (self.soft_max - self.max_area)
        
        return size_diff + 0.5 * boundary_penalty


class CombinedLoss(nn.Module):
    """Combined Dice + BCE + Boundary + AJI loss with class imbalance handling."""
    
    def __init__(self, pos_weight=10.0, boundary_weight=0.3, aji_weight=0.2, 
                 use_boundary=True, use_aji=True):
        super().__init__()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(boundary_width=3)
        self.aji = AJILoss()
        self.pos_weight = pos_weight
        self.boundary_weight = boundary_weight
        self.aji_weight = aji_weight
        self.use_boundary = use_boundary
        self.use_aji = use_aji

    def forward(self, pred, target, box=None):
        """
        Compute loss within bounding box region to handle class imbalance.
        
        Args:
            pred: (H, W) prediction logits
            target: (H, W) binary ground truth
            box: [x1, y1, x2, y2] bounding box (optional)
        """
        if box is not None:
            x1, y1, x2, y2 = box
            h, w = pred.shape[-2:]
            bw, bh = x2 - x1, y2 - y1
            expand = 0.1  # Reduced from 0.2 to fix over-segmentation
            x1 = max(0, int(x1 - bw * expand))
            y1 = max(0, int(y1 - bh * expand))
            x2 = min(w, int(x2 + bw * expand))
            y2 = min(h, int(y2 + bh * expand))

            pred_box = pred[..., y1:y2, x1:x2]
            target_box = target[..., y1:y2, x1:x2]
        else:
            pred_box = pred
            target_box = target

        n_pos = target_box.sum()
        n_neg = target_box.numel() - n_pos
        if n_pos > 0:
            dyn_pos_weight = min(n_neg / n_pos, self.pos_weight)
        else:
            dyn_pos_weight = self.pos_weight

        pos_weight_tensor = torch.as_tensor(dyn_pos_weight, dtype=pred.dtype, device=pred.device)
        bce = F.binary_cross_entropy_with_logits(
            pred_box.reshape(-1),
            target_box.reshape(-1).float(),
            pos_weight=pos_weight_tensor
        )

        pred_sigmoid = torch.sigmoid(pred_box)
        dice = self.dice(pred_sigmoid, target_box)

        base_loss = 0.5 * dice + 0.5 * bce
        
        # Calculate additional loss weights
        remaining_weight = 1.0
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Boundary loss
        if self.use_boundary and n_pos > 0:
            try:
                boundary_loss = self.boundary(pred_sigmoid, target_box)
                total_loss = total_loss + self.boundary_weight * boundary_loss
                remaining_weight -= self.boundary_weight
            except Exception:
                pass
        
        # AJI loss
        if self.use_aji and n_pos > 0:
            try:
                aji_loss = self.aji(pred_sigmoid, target_box)
                total_loss = total_loss + self.aji_weight * aji_loss
                remaining_weight -= self.aji_weight
            except Exception:
                pass
        
        # Base loss (Dice + BCE)
        total_loss = total_loss + remaining_weight * base_loss

        return total_loss

