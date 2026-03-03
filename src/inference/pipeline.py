"""
[DEPRECATED] Legacy inference pipeline — replaced by inference/core.py

⚠️ This file still uses model.model.* (Stage 1 weights) and the old
sam_preprocess() pipeline. It is NOT compatible with Plan B (model_cp + official pipeline).
Use inference/core.py segment_with_boxes() instead.

Original description:
Unified inference pipeline for CellSAM.
This module provides the main entry point for running inference,
consolidating all post-processing and visualization into a single interface.
"""
import warnings
warnings.warn(
    "inference/pipeline.py is deprecated. Use inference/core.py segment_with_boxes() instead.",
    DeprecationWarning, stacklevel=2
)
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform as skt

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cellSAM_source"))
from cellSAM import get_model

from .postprocess import smooth_boundary, keep_largest_component, validate_cell_size
from .visualize import mask_to_rgb, create_overlay


def normalize_channel(img: np.ndarray) -> np.ndarray:
    """Normalize image using percentile-based normalization."""
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        return np.clip((img - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(img, dtype=np.float32)


def load_model(checkpoint_path: str = None, device: str = "cuda") -> torch.nn.Module:
    """
    Load CellSAM model with optional checkpoint.
    
    Args:
        checkpoint_path: Path to fine-tuned checkpoint (optional)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    model = get_model()
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    model = model.to(device).eval()
    return model


def run_sam_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    boxes: list,
    device: str = "cuda",
    target_size: tuple = (1024, 1024),
    apply_postprocess: bool = True,
    validate_size: bool = True
) -> np.ndarray:
    """
    Run SAM inference on an image with bounding boxes.
    
    Args:
        model: CellSAM model
        image: Input image (H, W) or (H, W, 3)
        boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        device: Device for inference
        target_size: Size to resize image for SAM (default 1024x1024)
        apply_postprocess: Whether to apply boundary smoothing
        validate_size: Whether to validate cell sizes
    
    Returns:
        Instance segmentation mask with cell IDs
    """
    # Prepare image
    if image.ndim == 2:
        img_norm = normalize_channel(image)
        img_resized = skt.resize(img_norm, target_size, preserve_range=True)
        img_3ch = np.stack([img_resized] * 3, axis=-1)
    else:
        img_resized = skt.resize(image, (*target_size, 3), preserve_range=True)
        img_3ch = img_resized
    
    img_tensor = torch.from_numpy(img_3ch * 255).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Scale boxes
    h, w = image.shape[:2]
    scale_y, scale_x = target_size[0] / h, target_size[1] / w
    scaled_boxes = [[b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y] for b in boxes]
    
    # Run inference
    instance_mask = np.zeros(target_size, dtype=np.int32)
    cell_id = 0
    
    with torch.no_grad():
        img_preprocessed = model.sam_preprocess(img_tensor)
        image_embedding = model.model.image_encoder(img_preprocessed)
        
        for box in scaled_boxes:
            box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
            
            try:
                sparse_emb, dense_emb = model.model.prompt_encoder(
                    points=None, boxes=box_tensor, masks=None
                )
                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                
                pred = F.interpolate(low_res_masks, size=target_size, 
                                    mode="bilinear", align_corners=False)
                pred_binary = (torch.sigmoid(pred) > 0.5).cpu().numpy()[0, 0].astype(bool)
                
                # Post-processing
                if apply_postprocess:
                    pred_binary = smooth_boundary(pred_binary)
                
                pred_binary = keep_largest_component(pred_binary)
                
                # Size validation
                if validate_size and not validate_cell_size(pred_binary):
                    continue
                
                if pred_binary.sum() > 0:
                    cell_id += 1
                    # Only assign to empty pixels
                    new_pixels = pred_binary & (instance_mask == 0)
                    instance_mask[new_pixels] = cell_id
                    
            except Exception as e:
                print(f"Box inference failed: {e}")
                continue
    
    # Resize back to original size
    instance_resized = skt.resize(instance_mask, (h, w), order=0, 
                                   preserve_range=True).astype(np.int32)
    return instance_resized


def visualize_results(
    bf_image: np.ndarray,
    gt_mask: np.ndarray = None,
    pred_mask: np.ndarray = None,
    save_path: str = None,
    show: bool = False
) -> dict:
    """
    Visualize inference results.
    
    Args:
        bf_image: Brightfield image
        gt_mask: Ground truth instance mask (optional)
        pred_mask: Predicted instance mask (optional)
        save_path: Path to save visualization (optional)
        show: Whether to display using matplotlib
    
    Returns:
        Dict containing RGB visualizations
    """
    results = {'bf': bf_image}
    
    if gt_mask is not None:
        results['gt_rgb'] = mask_to_rgb(gt_mask)
        results['gt_overlay'] = create_overlay(bf_image, gt_mask)
    
    if pred_mask is not None:
        results['pred_rgb'] = mask_to_rgb(pred_mask)
        results['pred_overlay'] = create_overlay(bf_image, pred_mask)
    
    if save_path:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n_plots = 1 + (gt_mask is not None) + (pred_mask is not None)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        idx = 0
        axes[idx].imshow(bf_image, cmap='gray')
        axes[idx].set_title('Brightfield')
        axes[idx].axis('off')
        idx += 1
        
        if gt_mask is not None:
            axes[idx].imshow(results['gt_overlay'])
            axes[idx].set_title(f'GT ({gt_mask.max()} cells)')
            axes[idx].axis('off')
            idx += 1
        
        if pred_mask is not None:
            axes[idx].imshow(results['pred_overlay'])
            axes[idx].set_title(f'Pred ({pred_mask.max()} cells)')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        # Similar plotting code...
        plt.show()
    
    return results
