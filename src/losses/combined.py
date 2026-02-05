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
    GPU-accelerated Boundary Loss for cell segmentation (Updated 2026-02-05).
    
    Focuses training on pixels near cell edges using pure PyTorch operations.
    ~100x faster than scipy-based version.
    
    Reference: Kervadec et al., "Boundary loss for highly unbalanced segmentation"
    """
    
    def __init__(self, boundary_width=3):
        super().__init__()
        self.boundary_width = boundary_width
        self.kernel_size = boundary_width * 2 + 1
    
    def _gpu_erosion(self, mask: torch.Tensor) -> torch.Tensor:
        """GPU-based morphological erosion using max pooling."""
        # Ensure 4D tensor (B, C, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Erosion = min pooling = 1 - max_pool(1 - mask)
        inverted = 1 - mask
        dilated = F.max_pool2d(
            inverted, 
            self.kernel_size, 
            stride=1, 
            padding=self.kernel_size // 2
        )
        eroded = 1 - dilated
        
        return eroded.squeeze()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-focused loss using GPU operations.
        
        Args:
            pred: (H, W) or (B, H, W) prediction probabilities (after sigmoid)
            target: (H, W) or (B, H, W) binary ground truth
        """
        # Ensure float
        target_float = target.float()
        
        # GPU erosion
        eroded = self._gpu_erosion(target_float)
        
        # Boundary = original - eroded
        boundary = target_float - eroded
        boundary = (boundary > 0).float()
        
        n_boundary = boundary.sum()
        
        if n_boundary > 10:  # Minimum boundary pixels
            # Extract boundary pixels
            boundary_pred = pred[boundary > 0]
            boundary_target = target_float[boundary > 0]
            
            # BCE on boundary
            boundary_bce = F.binary_cross_entropy(
                boundary_pred.clamp(1e-7, 1 - 1e-7),
                boundary_target,
                reduction='mean'
            )
            
            # Dice on boundary
            intersection = (boundary_pred * boundary_target).sum()
            boundary_dice = 1 - (2. * intersection + 1) / (boundary_pred.sum() + boundary_target.sum() + 1)
            
            return 0.5 * boundary_bce + 0.5 * boundary_dice
        else:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)


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


class ContourLoss(nn.Module):
    """
    Contour distance loss for cell segmentation (2026-02-05).
    
    Computes distance transform based loss to penalize boundary errors.
    Reference: boundary_enhancement_design.md section 1.2
    """
    
    def __init__(self, boundary_width: int = 3):
        """
        Args:
            boundary_width: Width of boundary region to focus on
        """
        super().__init__()
        self.boundary_width = boundary_width
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute contour distance loss.
        
        Args:
            pred: Prediction probabilities (after sigmoid), shape (H, W)
            target: Binary ground truth, shape (H, W)
        
        Returns:
            Contour loss value
        """
        from scipy.ndimage import distance_transform_edt
        from skimage import morphology
        
        device = pred.device
        dtype = pred.dtype
        
        pred_np = (pred > 0.5).float().detach().cpu().numpy()
        target_np = (target > 0).float().cpu().numpy()
        
        # Extract boundaries using erosion
        struct = morphology.disk(1)
        pred_boundary = pred_np - morphology.binary_erosion(pred_np, struct).astype(np.float32)
        gt_boundary = target_np - morphology.binary_erosion(target_np, struct).astype(np.float32)
        
        # Compute distance transforms
        if gt_boundary.sum() > 0:
            gt_dist = distance_transform_edt(1 - gt_boundary)
        else:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        if pred_boundary.sum() > 0:
            pred_dist = distance_transform_edt(1 - pred_boundary)
        else:
            return torch.tensor(1.0, device=device, dtype=dtype)
        
        # Average distance of pred boundary to GT boundary
        pred_to_gt = gt_dist[pred_boundary > 0].mean() if pred_boundary.sum() > 0 else 0
        gt_to_pred = pred_dist[gt_boundary > 0].mean() if gt_boundary.sum() > 0 else 0
        
        # Symmetric Hausdorff-like loss (normalized)
        max_dist = max(gt_dist.max(), pred_dist.max(), 1.0)
        loss = (pred_to_gt + gt_to_pred) / (2 * max_dist)
        
        return torch.tensor(loss, device=device, dtype=dtype)


class CombinedLoss(nn.Module):
    """
    Combined loss for instance segmentation (Updated 2026-02-05).
    
    Supports:
    - Dice Loss (default)
    - BCE Loss (default)
    - Boundary Loss (configurable)
    - AJI Loss (configurable)
    - Topology Loss (configurable) - penalizes fragments
    - Size Loss (configurable) - penalizes size deviations
    - Contour Loss (configurable) - penalizes boundary distance errors
    """
    
    def __init__(self, pos_weight=10.0, boundary_weight=0.3, aji_weight=0.2, 
                 use_boundary=True, use_aji=True,
                 use_topology=False, topology_weight=0.1,
                 use_size=False, size_weight=0.1,
                 use_contour=False, contour_weight=0.1):
        """
        Args:
            pos_weight: Weight for positive class in BCE
            boundary_weight: Weight for boundary loss
            aji_weight: Weight for AJI loss
            use_boundary: Whether to use boundary loss
            use_aji: Whether to use AJI loss
            use_topology: Whether to use topology loss (Phase 2)
            topology_weight: Weight for topology loss
            use_size: Whether to use size loss (Phase 2)
            size_weight: Weight for size loss
            use_contour: Whether to use contour loss (Phase 2)
            contour_weight: Weight for contour loss
        """
        super().__init__()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(boundary_width=3)
        self.aji = AJILoss()
        self.topology = TopologyLoss()
        self.size = SizeLoss()
        self.contour = ContourLoss()
        
        self.pos_weight = pos_weight
        self.boundary_weight = boundary_weight
        self.aji_weight = aji_weight
        self.topology_weight = topology_weight
        self.size_weight = size_weight
        self.contour_weight = contour_weight
        
        self.use_boundary = use_boundary
        self.use_aji = use_aji
        self.use_topology = use_topology
        self.use_size = use_size
        self.use_contour = use_contour

    def forward(self, pred, target, box=None):
        """
        Compute combined loss within bounding box region.
        
        Args:
            pred: (H, W) prediction logits
            target: (H, W) binary ground truth
            box: [x1, y1, x2, y2] bounding box (optional)
        """
        if box is not None:
            x1, y1, x2, y2 = box
            h, w = pred.shape[-2:]
            bw, bh = x2 - x1, y2 - y1
            expand = 0.1
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
        
        # Calculate total weights
        total_extra_weight = 0.0
        if self.use_boundary:
            total_extra_weight += self.boundary_weight
        if self.use_aji:
            total_extra_weight += self.aji_weight
        if self.use_topology:
            total_extra_weight += self.topology_weight
        if self.use_size:
            total_extra_weight += self.size_weight
        if self.use_contour:
            total_extra_weight += self.contour_weight
        
        base_weight = max(0.3, 1.0 - total_extra_weight)  # Minimum 30% for base loss
        
        total_loss = base_weight * base_loss
        
        # Add optional losses
        if self.use_boundary and n_pos > 0:
            try:
                boundary_loss = self.boundary(pred_sigmoid, target_box)
                total_loss = total_loss + self.boundary_weight * boundary_loss
            except Exception:
                pass
        
        if self.use_aji and n_pos > 0:
            try:
                aji_loss = self.aji(pred_sigmoid, target_box)
                total_loss = total_loss + self.aji_weight * aji_loss
            except Exception:
                pass
        
        if self.use_topology and n_pos > 0:
            try:
                topo_loss = self.topology(pred_sigmoid, target_box)
                total_loss = total_loss + self.topology_weight * topo_loss
            except Exception:
                pass
        
        if self.use_size and n_pos > 0:
            try:
                size_loss = self.size(pred_sigmoid, target_box)
                total_loss = total_loss + self.size_weight * size_loss
            except Exception:
                pass
        
        if self.use_contour and n_pos > 0:
            try:
                contour_loss = self.contour(pred_sigmoid, target_box)
                total_loss = total_loss + self.contour_weight * contour_loss
            except Exception:
                pass

        return total_loss


