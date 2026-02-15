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
    Differentiable topology loss for cell segmentation (GPU).
    
    Penalizes small prediction fragments using morphological opening.
    opening = dilation(erosion(pred)) removes small objects.
    fragments = pred - opening(pred) = pixels that shouldn't exist.
    
    Fully differentiable: uses max-pool based morphology (no numpy/scipy).
    
    Replaces: old scipy.ndimage version (zero gradient, 2026-02-05).
    Updated: 2026-02-12 (Phase 2 Step 2).
    """
    
    def __init__(self, min_radius: int = 3):
        """
        Args:
            min_radius: Objects smaller than this radius are considered fragments.
                        Kernel size = 2 * min_radius + 1.
        """
        super().__init__()
        self.min_radius = min_radius
        self.kernel_size = 2 * min_radius + 1
    
    def _ensure_4d(self, x):
        """Ensure tensor is (B, C, H, W) for max_pool2d."""
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            return x.unsqueeze(0)
        return x
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """
        Compute topology loss.
        
        Args:
            pred: (H, W) prediction probabilities (after sigmoid)
            target: Not used, for API consistency
        
        Returns:
            Scalar loss — mean of fragment pixels
        """
        p = self._ensure_4d(pred)
        pad = self.kernel_size // 2
        
        # Morphological erosion: min_pool = 1 - max_pool(1 - p)
        eroded = 1.0 - F.max_pool2d(1.0 - p, self.kernel_size, stride=1, padding=pad)
        
        # Morphological dilation: max_pool on eroded
        opened = F.max_pool2d(eroded, self.kernel_size, stride=1, padding=pad)
        
        # Fragment mask: pred pixels removed by opening
        fragments = (pred - opened.squeeze()).clamp(0, 1)
        
        return fragments.mean()


class SizeLoss(nn.Module):
    """
    Size constraint loss for cell segmentation.
    
    Penalizes predictions that deviate from target cell size,
    encouraging the model to learn correct cell boundaries.
    
    Based on GT analysis (FULL dataset: 478 images, 5173 cells):
    Original (1736×1776): P1=40836, P99=513928, Median=142316
    Scaled (1024px): P1=13884, P99=174735, Median=48387 (×0.340)
    """
    
    def __init__(self, min_area: int = 13884, max_area: int = 174735, 
                 smooth: float = 1.0, margin: float = 0.2):
        """
        Args:
            min_area: Minimum expected cell area (E17 P1 scaled to 1024)
            max_area: Maximum expected cell area (E17 P99 scaled to 1024)
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
    Differentiable contour distance loss for cell segmentation (GPU).
    
    Uses max-pool erosion (same as BoundaryLoss) to extract boundaries,
    then iterative dilation to approximate distance transform.
    Penalizes pred pixels far from GT boundary + missing GT boundary pixels.
    
    Fully differentiable: pure PyTorch operations (no numpy/scipy).
    
    Replaces: old scipy.ndimage.distance_transform_edt version (zero gradient).
    Updated: 2026-02-12 (Phase 2 Step 2).
    """
    
    def __init__(self, boundary_width: int = 3, n_distance_steps: int = 5):
        """
        Args:
            boundary_width: Erosion kernel radius for boundary extraction
            n_distance_steps: Number of dilation steps for distance approximation
        """
        super().__init__()
        self.boundary_width = boundary_width
        self.kernel_size = 2 * boundary_width + 1
        self.n_distance_steps = n_distance_steps
    
    def _ensure_4d(self, x):
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            return x.unsqueeze(0)
        return x
    
    def _extract_boundary(self, mask):
        """Extract boundary via GPU erosion: boundary = mask - erode(mask)."""
        m = self._ensure_4d(mask.float())
        pad = self.kernel_size // 2
        eroded = 1.0 - F.max_pool2d(1.0 - m, self.kernel_size, stride=1, padding=pad)
        boundary = (m - eroded).squeeze().clamp(0, 1)
        return boundary
    
    def _approx_distance(self, boundary):
        """
        Approximate distance transform via iterative 3x3 dilation.
        Returns distance map where each pixel = distance to nearest boundary pixel.
        """
        b = self._ensure_4d(boundary)
        distance = torch.zeros_like(b)
        reached = (b > 0.5).float()
        
        for step in range(1, self.n_distance_steps + 1):
            # Dilate the reached region
            dilated = F.max_pool2d(reached, 3, stride=1, padding=1)
            new_pixels = (dilated - reached).clamp(0, 1)
            distance = distance + step * new_pixels
            reached = dilated
        
        return distance.squeeze()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute contour distance loss.
        
        Args:
            pred: (H, W) prediction probabilities (after sigmoid)
            target: (H, W) binary ground truth
        
        Returns:
            Scalar contour loss
        """
        gt_boundary = self._extract_boundary(target)
        
        if gt_boundary.sum() < 5:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Approximate distance from GT boundary
        gt_dist = self._approx_distance(gt_boundary)
        
        # Normalize distance to [0, 1]
        max_dist = gt_dist.max().clamp(min=1.0)
        gt_dist_norm = gt_dist / max_dist
        
        # Loss 1: Penalize pred=high far from GT boundary
        # pred pixels weighted by their distance to GT boundary
        far_penalty = (pred * gt_dist_norm).mean()
        
        # Loss 2: Penalize missing GT boundary pixels
        miss_penalty = ((1 - pred) * gt_boundary).mean()
        
        return far_penalty + miss_penalty


class NeighborIntrusionLoss(nn.Module):
    """
    Neighbor intrusion loss (L_neighbor) — Phase 2 Step 3.
    
    Penalizes predicting high confidence in neighboring cell GT regions.
    Works per-box (needs instance_mask).
    
    Formula: L_neighbor(k) = mean(n_k * p_k^gamma)
    where n_k = neighbor GT pixels, p_k = prediction probability.
    """
    
    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                instance_mask: torch.Tensor = None) -> torch.Tensor:
        if instance_mask is None:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Neighbor region: other cells (not current, not background)
        cell_region = target > 0.5
        neighbor_region = (instance_mask > 0) & (~cell_region)
        neighbor_mask = neighbor_region.float()
        
        n_neighbor = neighbor_mask.sum()
        if n_neighbor < 1:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        intrusion = neighbor_mask * (pred ** self.gamma)
        return intrusion.sum() / (n_neighbor + 1e-6)


class OverlapMutexLoss(nn.Module):
    """
    Overlap mutex loss (L_overlap) — Phase 2 Step 3.
    
    Prevents same pixel from being claimed by multiple cells.
    Single-pass approximation: confidence_map (detached) holds prior boxes.
    
    Formula: L_overlap = mean(ReLU(S - 1 - margin)^2)
    where S(x) = confidence_map(x) + pred(x).
    """
    
    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin
    
    def forward(self, pred: torch.Tensor,
                confidence_map: torch.Tensor = None) -> torch.Tensor:
        if confidence_map is None:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        local_sum = confidence_map + pred
        excess = F.relu(local_sum - 1.0 - self.margin)
        return (excess ** 2).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for instance segmentation.
    
    Updated 2026-02-12 (Phase 2):
    - All losses fully differentiable (GPU)
    - Normalized weights (sum to 1.0)
    - L_neighbor + L_overlap support
    """
    
    def __init__(self, pos_weight=10.0, boundary_weight=0.3, aji_weight=0.2, 
                 use_boundary=True, use_aji=True,
                 use_topology=False, topology_weight=0.1,
                 use_size=False, size_weight=0.1,
                 use_contour=False, contour_weight=0.1,
                 use_neighbor=False, neighbor_weight=0.3,
                 use_overlap=False, overlap_weight=0.1,
                 neighbor_gamma=1.5, overlap_margin=0.05,
                 delay_epochs=0, ramp_epochs=10):
        super().__init__()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(boundary_width=3)
        self.aji = AJILoss()
        self.topology = TopologyLoss()
        self.size = SizeLoss()
        self.contour = ContourLoss()
        self.neighbor_loss_fn = NeighborIntrusionLoss(gamma=neighbor_gamma)
        self.overlap_loss_fn = OverlapMutexLoss(margin=overlap_margin)
        
        self.pos_weight = pos_weight
        self.boundary_weight = boundary_weight
        self.aji_weight = aji_weight
        self.topology_weight = topology_weight
        self.size_weight = size_weight
        self.contour_weight = contour_weight
        # Fix3: Store base weights for delayed enable (phase2_design.md §8.4)
        self._base_neighbor_weight = neighbor_weight
        self._base_overlap_weight = overlap_weight
        self.neighbor_weight = neighbor_weight
        self.overlap_weight = overlap_weight
        self.delay_epochs = delay_epochs
        self.ramp_epochs = ramp_epochs
        self._current_epoch = 0
        
        self.use_boundary = use_boundary
        self.use_aji = use_aji
        self.use_topology = use_topology
        self.use_size = use_size
        self.use_contour = use_contour
        self.use_neighbor = use_neighbor
        self.use_overlap = use_overlap

    def set_epoch(self, epoch):
        """Update N/O weights based on epoch for delayed enable + linear ramp.
        
        Fix3 schedule (phase2_design.md §8.4):
          epoch < delay_epochs       → N/O weight = 0 (pure P1 losses)
          delay ≤ epoch < delay+ramp → linear ramp from 0 to base weight
          epoch ≥ delay+ramp         → full base weight
        """
        self._current_epoch = epoch
        if epoch < self.delay_epochs:
            scale = 0.0
        elif self.ramp_epochs > 0:
            scale = min(1.0, (epoch - self.delay_epochs) / self.ramp_epochs)
        else:
            scale = 1.0
        self.neighbor_weight = self._base_neighbor_weight * scale
        self.overlap_weight = self._base_overlap_weight * scale

    def forward(self, pred, target, box=None,
                instance_mask=None, confidence_map=None):
        """
        Compute combined loss within bounding box region.
        
        Args:
            pred: (H, W) prediction logits
            target: (H, W) binary ground truth
            box: [x1, y1, x2, y2] bounding box (optional)
            instance_mask: (H, W) full instance mask for L_neighbor (Phase 2)
            confidence_map: (H, W) accumulated predictions for L_overlap (Phase 2)
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
            
            # Clip instance_mask and confidence_map to box
            instance_mask_box = instance_mask[..., y1:y2, x1:x2] if instance_mask is not None else None
            confidence_map_box = confidence_map[..., y1:y2, x1:x2] if confidence_map is not None else None
        else:
            pred_box = pred
            target_box = target
            instance_mask_box = instance_mask
            confidence_map_box = confidence_map

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
        
        # Normalized weight computation
        # Computability-gated: only include weight in denominator if the loss
        # CAN be computed (required inputs are present). This prevents silent
        # loss scale reduction when instance_mask or confidence_map is None.
        # (Codex finding 17.9.2, 2026-02-13)
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
        # Neighbor/overlap: only count weight if input is computable
        # Check both presence AND shape compatibility (Codex 17.10 finding #2)
        neighbor_computable = (
            self.use_neighbor
            and instance_mask_box is not None
            and instance_mask_box.shape[-2:] == pred_box.shape[-2:]
        )
        overlap_computable = (
            self.use_overlap
            and confidence_map_box is not None
            and confidence_map_box.shape[-2:] == pred_box.shape[-2:]
        )
        if neighbor_computable:
            total_extra_weight += self.neighbor_weight
        if overlap_computable:
            total_extra_weight += self.overlap_weight
        
        raw_base = 0.3
        total_weight = raw_base + total_extra_weight
        
        total_loss = (raw_base / total_weight) * base_loss
        
        # Add optional losses
        if self.use_boundary and n_pos > 0:
            try:
                boundary_loss = self.boundary(pred_sigmoid, target_box)
                total_loss = total_loss + (self.boundary_weight / total_weight) * boundary_loss
            except Exception as e:
                import warnings
                warnings.warn(f"BoundaryLoss failed: {e}")
        
        if self.use_aji and n_pos > 0:
            try:
                aji_loss = self.aji(pred_sigmoid, target_box)
                total_loss = total_loss + (self.aji_weight / total_weight) * aji_loss
            except Exception as e:
                import warnings
                warnings.warn(f"AJILoss failed: {e}")
        
        if self.use_topology and n_pos > 0:
            try:
                topo_loss = self.topology(pred_sigmoid, target_box)
                total_loss = total_loss + (self.topology_weight / total_weight) * topo_loss
            except Exception as e:
                import warnings
                warnings.warn(f"TopologyLoss failed: {e}")
        
        if self.use_size and n_pos > 0:
            try:
                size_loss = self.size(pred_sigmoid, target_box)
                total_loss = total_loss + (self.size_weight / total_weight) * size_loss
            except Exception as e:
                import warnings
                warnings.warn(f"SizeLoss failed: {e}")
        
        if self.use_contour and n_pos > 0:
            try:
                contour_loss = self.contour(pred_sigmoid, target_box)
                total_loss = total_loss + (self.contour_weight / total_weight) * contour_loss
            except Exception as e:
                import warnings
                warnings.warn(f"ContourLoss failed: {e}")
        
        if neighbor_computable and n_pos > 0:
            try:
                neighbor_loss = self.neighbor_loss_fn(pred_sigmoid, target_box, instance_mask_box)
                total_loss = total_loss + (self.neighbor_weight / total_weight) * neighbor_loss
            except Exception as e:
                import warnings
                warnings.warn(f"NeighborIntrusionLoss failed: {e}")
        
        if overlap_computable:
            try:
                overlap_loss = self.overlap_loss_fn(pred_sigmoid, confidence_map_box)
                total_loss = total_loss + (self.overlap_weight / total_weight) * overlap_loss
            except Exception as e:
                import warnings
                warnings.warn(f"OverlapMutexLoss failed: {e}")

        return total_loss






