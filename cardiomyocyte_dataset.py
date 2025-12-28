"""
PyTorch Dataset for Cardiomyocyte CellSAM Training.
Loads brightfield images, DAPI-derived prompts, and ground truth segmentation masks.

Usage:
    from cardiomyocyte_dataset import CardiomyocyteDataset
    dataset = CardiomyocyteDataset(data_dir="path/to/processed_data")
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from skimage import transform


class CardiomyocyteDataset(Dataset):
    """
    Dataset for training CellSAM on cardiomyocyte images.
    
    Expected directory structure:
        data_dir/
        ├── images/
        │   ├── sample1_brightfield.npy
        │   ├── sample2_brightfield.npy
        │   └── ...
        ├── prompts/
        │   ├── sample1_dapi_prompts.json
        │   ├── sample2_dapi_prompts.json
        │   └── ...
        └── masks/
            ├── sample1_segmentation.npy
            ├── sample2_segmentation.npy
            └── ...
    """
    
    def __init__(
        self,
        data_dir: str,
        image_subdir: str = "images",
        prompt_subdir: str = "prompts",
        mask_subdir: str = "masks",
        image_pattern: str = "*brightfield*.npy",
        target_size: Tuple[int, int] = (1024, 1024),
        normalize: bool = True,
        max_boxes_per_image: int = 100,
        augment: bool = False
    ):
        """
        Args:
            data_dir: Root directory containing images, prompts, masks subdirs
            image_subdir: Subdirectory name for images
            prompt_subdir: Subdirectory name for prompt JSON files
            mask_subdir: Subdirectory name for segmentation masks
            image_pattern: Glob pattern to find image files
            target_size: Resize images to this size (H, W)
            normalize: Normalize images to 0-1 range
            max_boxes_per_image: Maximum number of bounding boxes to use per image
            augment: Apply data augmentation (not implemented yet)
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.normalize = normalize
        self.max_boxes = max_boxes_per_image
        self.augment = augment
        
        # Collect samples
        image_dir = self.data_dir / image_subdir
        prompt_dir = self.data_dir / prompt_subdir
        mask_dir = self.data_dir / mask_subdir
        
        self.samples = []
        
        for image_path in image_dir.glob(image_pattern):
            # Extract sample ID from filename (assumes format: sampleID_brightfield.npy)
            sample_id = image_path.stem.replace("_brightfield", "").replace("_channel_0", "")
            
            # Find corresponding prompt file
            prompt_candidates = list(prompt_dir.glob(f"*{sample_id}*_prompts.json"))
            if not prompt_candidates:
                print(f"Warning: No prompt file found for {sample_id}")
                continue
            prompt_path = prompt_candidates[0]
            
            # Find corresponding mask file
            mask_candidates = list(mask_dir.glob(f"*{sample_id}*.npy"))
            if not mask_candidates:
                print(f"Warning: No mask file found for {sample_id}")
                continue
            mask_path = mask_candidates[0]
            
            self.samples.append({
                "sample_id": sample_id,
                "image_path": str(image_path),
                "prompt_path": str(prompt_path),
                "mask_path": str(mask_path)
            })
        
        print(f"Found {len(self.samples)} valid samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_and_resize(self, path: str, is_mask: bool = False) -> np.ndarray:
        """Load numpy array and resize to target size."""
        arr = np.load(path)
        
        if arr.shape[:2] != self.target_size:
            order = 0 if is_mask else 1  # Nearest for masks, bilinear for images
            arr = transform.resize(
                arr, 
                self.target_size, 
                order=order,
                preserve_range=True,
                anti_aliasing=not is_mask
            )
        
        return arr
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range."""
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0
        
        # Percentile normalization
        p_low, p_high = np.percentile(img, [1, 99])
        img = np.clip(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + 1e-8)
        
        return img.astype(np.float32)
    
    def _format_boxes_for_sam(
        self, 
        boxes: List[Dict], 
        original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Convert bounding boxes to SAM format.
        SAM expects boxes in (x1, y1, x2, y2) format, normalized or absolute.
        
        Args:
            boxes: List of box dicts with 'box' key
            original_size: Original image size (H, W) for scaling
        
        Returns:
            Tensor of shape (N, 4) with box coordinates
        """
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        
        orig_h, orig_w = original_size
        target_h, target_w = self.target_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        formatted = []
        for box_data in boxes[:self.max_boxes]:
            x1, y1, x2, y2 = box_data['box']
            # Scale to target size
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            formatted.append([x1, y1, x2, y2])
        
        return torch.tensor(formatted, dtype=torch.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
                - 'image': (3, H, W) tensor (RGB channels, same value for grayscale)
                - 'boxes': (N, 4) tensor of bounding boxes
                - 'mask': (H, W) tensor of ground truth segmentation
                - 'sample_id': string identifier
        """
        sample = self.samples[idx]
        
        # Load image
        image = self._load_and_resize(sample['image_path'], is_mask=False)
        
        if self.normalize:
            image = self._normalize_image(image)
        
        # Convert grayscale to RGB (CellSAM expects 3 channels)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=0)
        elif image.ndim == 3 and image.shape[-1] == 3:
            image = image.transpose(2, 0, 1)  # HWC -> CHW
        else:
            image = np.stack([image, image, image], axis=0)
        
        # Load prompts
        with open(sample['prompt_path'], 'r') as f:
            prompt_data = json.load(f)
        
        original_size = tuple(prompt_data.get('image_shape', self.target_size))
        boxes = self._format_boxes_for_sam(prompt_data['boxes'], original_size)
        
        # Load mask
        mask = self._load_and_resize(sample['mask_path'], is_mask=True)
        mask = mask.astype(np.int64)
        
        return {
            'image': torch.from_numpy(image).float(),
            'boxes': boxes,
            'mask': torch.from_numpy(mask).long(),
            'sample_id': sample['sample_id'],
            'num_boxes': len(prompt_data['boxes'])
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable number of boxes per image.
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    # Boxes need padding since different images have different numbers of boxes
    max_boxes = max(item['num_boxes'] for item in batch)
    
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
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    dataset = CardiomyocyteDataset(args.data_dir)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample ID: {sample['sample_id']}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Boxes shape: {sample['boxes'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Mask unique values: {torch.unique(sample['mask'])[:10]}")
    else:
        print("No samples found!")
