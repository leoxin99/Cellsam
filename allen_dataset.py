"""
Simplified Dataset for CellSAM training using extracted Allen Cell training pairs.
Uses instance segmentation masks to generate bounding box prompts.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple
from skimage import transform, measure


class AllenCellDataset(Dataset):
    """
    Dataset for training CellSAM using Allen Cell extracted data.
    
    Expected directory structure:
        training_pairs/
        ├── images/
        │   ├── sample1.npy
        │   └── ...
        └── masks/
            ├── sample1.npy
            └── ...
    
    Automatically generates bounding box prompts from instance masks.
    """
    
    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int] = (1024, 1024),
        normalize: bool = True,
        max_boxes_per_image: int = 50,
        augment: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.normalize = normalize
        self.max_boxes = max_boxes_per_image
        self.augment = augment
        
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        
        # Find all image files
        self.samples = []
        
        for image_path in self.image_dir.glob("*.npy"):
            sample_id = image_path.stem
            mask_path = self.mask_dir / f"{sample_id}.npy"
            
            if mask_path.exists():
                self.samples.append({
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path)
                })
            else:
                print(f"Warning: No mask found for {sample_id}")
        
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_and_resize(self, path: str, is_mask: bool = False) -> np.ndarray:
        """Load numpy array and resize to target size."""
        arr = np.load(path)
        
        if arr.shape[:2] != self.target_size:
            order = 0 if is_mask else 1
            arr = transform.resize(
                arr,
                self.target_size,
                order=order,
                preserve_range=True,
                anti_aliasing=not is_mask
            )
        
        return arr
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range with percentile normalization."""
        img = img.astype(np.float32)
        
        if img.max() > 1:
            img = img / 255.0
        
        p_low, p_high = np.percentile(img, [1, 99])
        img = np.clip(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + 1e-8)
        
        return img.astype(np.float32)
    
    def _mask_to_boxes(self, mask: np.ndarray) -> List[List[float]]:
        """
        Extract bounding boxes from instance segmentation mask.
        Each unique non-zero value is a separate instance.
        """
        boxes = []
        
        for region in measure.regionprops(mask.astype(np.int32)):
            # Get bounding box (min_row, min_col, max_row, max_col)
            y1, x1, y2, x2 = region.bbox
            
            # Add small padding
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(mask.shape[1], x2 + pad)
            y2 = min(mask.shape[0], y2 + pad)
            
            boxes.append([x1, y1, x2, y2])
        
        return boxes[:self.max_boxes]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = self._load_and_resize(sample['image_path'], is_mask=False)
        original_size = image.shape[:2]
        
        if self.normalize:
            image = self._normalize_image(image)
        
        # Convert to RGB (CellSAM expects 3 channels)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=0)
        elif image.ndim == 3 and image.shape[-1] == 3:
            image = image.transpose(2, 0, 1)
        else:
            image = np.stack([image, image, image], axis=0)
        
        # Load mask
        mask = self._load_and_resize(sample['mask_path'], is_mask=True)
        mask = mask.astype(np.int64)
        
        # Generate boxes from mask
        boxes = self._mask_to_boxes(mask)
        
        # Scale boxes to target size
        scale_y = self.target_size[0] / original_size[0]
        scale_x = self.target_size[1] / original_size[1]
        
        scaled_boxes = []
        for x1, y1, x2, y2 in boxes:
            scaled_boxes.append([
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ])
        
        boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32) if scaled_boxes else torch.zeros((0, 4))
        
        return {
            'image': torch.from_numpy(image).float(),
            'boxes': boxes_tensor,
            'mask': torch.from_numpy(mask).long(),
            'sample_id': sample['sample_id'],
            'num_boxes': len(scaled_boxes)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable number of boxes."""
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    max_boxes = max(item['num_boxes'] for item in batch) if batch else 0
    max_boxes = max(max_boxes, 1)  # Ensure at least 1 box slot
    
    padded_boxes = []
    box_counts = []
    
    for item in batch:
        boxes = item['boxes']
        num_boxes = boxes.shape[0]
        box_counts.append(num_boxes)
        
        if num_boxes < max_boxes:
            padding = torch.zeros((max_boxes - num_boxes, 4), dtype=torch.float32)
            boxes = torch.cat([boxes, padding], dim=0)
        
        padded_boxes.append(boxes)
    
    boxes = torch.stack(padded_boxes)
    
    return {
        'image': images,
        'boxes': boxes,
        'box_counts': torch.tensor(box_counts, dtype=torch.long),
        'mask': masks,
        'sample_ids': [item['sample_id'] for item in batch]
    }


if __name__ == "__main__":
    # Test the dataset
    data_dir = "d:/AI/paper/CellSam/training_pairs"
    
    dataset = AllenCellDataset(data_dir)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n=== Sample Test ===")
        print(f"Sample ID: {sample['sample_id']}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Boxes shape: {sample['boxes'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Number of boxes (cells): {sample['num_boxes']}")
        print(f"Mask unique values: {torch.unique(sample['mask'])}")
        
        # Test collate function
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"\n=== Batch Test ===")
        print(f"Batch images: {batch['image'].shape}")
        print(f"Batch boxes: {batch['boxes'].shape}")
        print(f"Batch masks: {batch['mask'].shape}")
        print(f"Box counts: {batch['box_counts']}")
