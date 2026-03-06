#!/usr/bin/env python3
"""CellFinder box detection visualization with napari.

Loads CellSAM model, runs CellFinder (generate_bounding_boxes) on test set,
and displays: DAPI, BF (if available), GT boxes, CellFinder boxes, GT masks.
"""
import sys, json
import numpy as np
import torch
import napari
from pathlib import Path
from skimage.measure import regionprops

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from cellSAM.model import get_local_model
from cellSAM.utils import format_image_shape, normalize_image


def load_test_samples(n=5):
    """Load first n test samples."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    ids = [l.strip() for l in open(splits_dir / "test_ids.txt") if l.strip()][:n]
    
    samples = []
    for img_id in ids:
        img = np.load(processed_dir / "images" / f"{img_id}.npy")
        mask = np.load(processed_dir / "masks" / f"{img_id}.npy")
        
        # Extract GT boxes
        gt_boxes = []
        for prop in regionprops(mask.astype(int)):
            y1, x1, y2, x2 = prop.bbox
            gt_boxes.append([x1, y1, x2, y2])
        
        samples.append({
            'id': img_id,
            'image': img,     # (C, H, W)
            'mask': mask,
            'gt_boxes': gt_boxes
        })
    
    return samples


def boxes_to_napari_rectangles(boxes, offset=0):
    """Convert [x1,y1,x2,y2] boxes to napari rectangle format [[y1,x1],[y2,x2]]."""
    rects = []
    for box in boxes:
        x1, y1, x2, y2 = box
        rects.append(np.array([
            [y1 + offset, x1],
            [y1 + offset, x2],
            [y2 + offset, x2],
            [y2 + offset, x1],
        ]))
    return rects


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading CellSAM model for CellFinder...")
    model_path = str(Path.home() / ".deepcell" / "models" / "cellsam_v1.2" / "cellsam_general.pt")
    model = get_local_model(model_path)
    model = model.to(device)
    model.eval()
    print("  Model loaded.")
    
    print("Loading test samples (first 5)...")
    samples = load_test_samples(n=5)
    print(f"  {len(samples)} samples loaded.")
    
    viewer = napari.Viewer(title="CellFinder Box Detection Test")
    
    for idx, s in enumerate(samples):
        print(f"\nProcessing sample {idx+1}/{len(samples)}: {s['id']}")
        
        img = s['image']  # (C, H, W)
        mask = s['mask']
        gt_boxes = s['gt_boxes']
        
        C, H, W = img.shape
        
        # Show DAPI and BF channels
        # Our processed images: channels may vary, typically BF is ch0 or last
        if C >= 2:
            dapi = img[1]  # DAPI usually channel 1
            bf = img[0]    # BF usually channel 0
        else:
            dapi = img[0]
            bf = img[0]
        
        y_offset = idx * (H + 20)
        
        viewer.add_image(dapi, name=f"[{idx}] DAPI", 
                        translate=[y_offset, 0],
                        colormap='blue', blending='additive', visible=True)
        viewer.add_image(bf, name=f"[{idx}] BF",
                        translate=[y_offset, 0],
                        colormap='gray', blending='additive', visible=True, opacity=0.5)
        viewer.add_labels(mask.astype(int), name=f"[{idx}] GT Mask",
                         translate=[y_offset, 0], opacity=0.3)
        
        # GT boxes
        gt_rects = boxes_to_napari_rectangles(gt_boxes, offset=y_offset)
        if gt_rects:
            viewer.add_shapes(gt_rects, shape_type='polygon',
                            edge_color='green', edge_width=2, face_color='transparent',
                            name=f"[{idx}] GT boxes ({len(gt_rects)})")
        
        # Run CellFinder
        # Prepare image for CellFinder: (H, W, C) or (H, W) -> normalize -> (C, H, W) -> torch
        img_hwc = img.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        img_formatted = format_image_shape(img_hwc)
        img_normalized = normalize_image(img_formatted)
        img_chw = img_normalized.transpose(2, 0, 1)  # back to (C,H,W)
        img_tensor = torch.from_numpy(img_chw).float().unsqueeze(0).to(device)
        
        print(f"  Running CellFinder...")
        with torch.no_grad():
            cf_boxes_list = model.generate_bounding_boxes(img_tensor, device=device)
        
        if cf_boxes_list and len(cf_boxes_list) > 0:
            cf_boxes_tensor = cf_boxes_list[0]
            if isinstance(cf_boxes_tensor, torch.Tensor):
                cf_boxes = cf_boxes_tensor.cpu().numpy().tolist()
            else:
                cf_boxes = cf_boxes_tensor.tolist() if hasattr(cf_boxes_tensor, 'tolist') else list(cf_boxes_tensor)
            
            print(f"  CellFinder detected {len(cf_boxes)} boxes (GT: {len(gt_boxes)})")
            
            cf_rects = boxes_to_napari_rectangles(cf_boxes, offset=y_offset)
            if cf_rects:
                viewer.add_shapes(cf_rects, shape_type='polygon',
                                edge_color='red', edge_width=2, face_color='transparent',
                                name=f"[{idx}] CellFinder boxes ({len(cf_rects)})")
        else:
            print(f"  CellFinder detected 0 boxes (GT: {len(gt_boxes)})")
        
        print(f"  GT: {len(gt_boxes)} boxes | CellFinder: {len(cf_boxes) if cf_boxes_list else 0} boxes")
    
    print("\n" + "=" * 60)
    print("Napari viewer opened. Green = GT boxes, Red = CellFinder boxes.")
    print("=" * 60)
    
    napari.run()


if __name__ == "__main__":
    main()
