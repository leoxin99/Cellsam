"""
Enhanced Dataset with Albumentations augmentation for CellSAM training.
Significantly increases effective training samples through geometric and intensity transforms.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple
from skimage import transform, measure

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Using basic transforms.")


def get_train_transforms(target_size=(1024, 1024)):
    """Get training augmentation transforms."""
    if not HAS_ALBUMENTATIONS:
        return None
    
    return A.Compose([
        # Geometric transforms
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=0,
            p=0.5
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=12,
            p=0.3
        ),
        
        # Intensity transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # Resize to target size
        A.Resize(height=target_size[0], width=target_size[1]),
    ])


def get_val_transforms(target_size=(1024, 1024)):
    """Get validation transforms (no augmentation)."""
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
    ])


def get_all_sample_ids(data_dir: str) -> List[str]:
    """Get all sample IDs from the processed data directory."""
    data_path = Path(data_dir)
    image_dir = data_path / "images"

    sample_ids = []
    for image_path in image_dir.glob("*.npy"):
        sample_ids.append(image_path.stem)

    return sorted(sample_ids)


def load_split_ids(split: str = "train", splits_dir: str = "d:/AI/paper/CellSam/data/splits") -> List[str]:
    """
    Load sample IDs from a fixed split file.
    
    Args:
        split: One of 'train', 'val', 'test'
        splits_dir: Directory containing the split ID files
    
    Returns:
        List of sample IDs for the specified split
    """
    split_file = Path(splits_dir) / f"{split}_ids.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(ids)} samples from {split} split")
    return ids


class AugmentedAllenDataset(Dataset):
    """
    Augmented dataset for CellSAM training with Albumentations.
    Supports both directory-based loading and explicit file list.
    """

    def __init__(
        self,
        data_dir: str = None,
        target_size: Tuple[int, int] = (1024, 1024),
        is_training: bool = True,
        max_boxes_per_image: int = 50,
        sample_ids: List[str] = None  # NEW: explicit sample ID list
    ):
        self.target_size = target_size
        self.max_boxes = max_boxes_per_image
        self.is_training = is_training

        # Setup transforms
        if is_training:
            self.transform = get_train_transforms(target_size)
        else:
            self.transform = get_val_transforms(target_size)

        # Build sample list
        self.samples = []

        if sample_ids is not None and data_dir is not None:
            # Use explicit sample ID list (for train/val split)
            data_path = Path(data_dir)
            image_dir = data_path / "images"
            mask_dir = data_path / "masks"

            for sample_id in sample_ids:
                image_path = image_dir / f"{sample_id}.npy"
                mask_path = mask_dir / f"{sample_id}.npy"

                if image_path.exists() and mask_path.exists():
                    self.samples.append({
                        "sample_id": sample_id,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path)
                    })
        elif data_dir is not None:
            # Scan directory for all samples
            data_path = Path(data_dir)
            image_dir = data_path / "images"
            mask_dir = data_path / "masks"

            for image_path in image_dir.glob("*.npy"):
                sample_id = image_path.stem
                mask_path = mask_dir / f"{sample_id}.npy"

                if mask_path.exists():
                    self.samples.append({
                        "sample_id": sample_id,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path)
                    })

        print(f"Loaded {len(self.samples)} samples (training={is_training})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Percentile normalization."""
        img = img.astype(np.float32)
        
        if img.max() > 1:
            img = img / 255.0
        
        p_low, p_high = np.percentile(img, [1, 99])
        img = np.clip(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + 1e-8)
        
        return img.astype(np.float32)
    
    def _augment_box(self, box: List[float], img_shape: Tuple[int, int], 
                     noise_ratio: float = 0.3) -> List[float]:
        """
        Apply random perturbation to bounding box.
        
        This helps the model learn to handle imprecise boxes during inference,
        where DAPI-detected boxes may be larger than the actual cell.
        
        Args:
            box: [x1, y1, x2, y2] bounding box
            img_shape: (height, width) of image
            noise_ratio: max ratio of box dimension to perturb (0.3 = 30%)
        
        Returns:
            Perturbed box [x1, y1, x2, y2]
        """
        import random
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Random expansion (more common) or contraction (less common)
        # Bias towards expansion since inference boxes are typically larger
        expand_x1 = random.uniform(0, w * noise_ratio)
        expand_y1 = random.uniform(0, h * noise_ratio)
        expand_x2 = random.uniform(0, w * noise_ratio)
        expand_y2 = random.uniform(0, h * noise_ratio)
        
        # 70% chance to expand, 30% chance to contract
        if random.random() > 0.3:
            x1_new = x1 - expand_x1
            y1_new = y1 - expand_y1
            x2_new = x2 + expand_x2
            y2_new = y2 + expand_y2
        else:
            x1_new = x1 + expand_x1 * 0.5
            y1_new = y1 + expand_y1 * 0.5
            x2_new = x2 - expand_x2 * 0.5
            y2_new = y2 - expand_y2 * 0.5
        
        # Clamp to image bounds
        x1_new = max(0, x1_new)
        y1_new = max(0, y1_new)
        x2_new = min(img_shape[1], x2_new)
        y2_new = min(img_shape[0], y2_new)
        
        # Ensure valid box
        if x2_new <= x1_new or y2_new <= y1_new:
            return box
        
        return [x1_new, y1_new, x2_new, y2_new]
    
    def _mask_to_boxes_with_ids(self, mask: np.ndarray, 
                                 apply_augment: bool = False) -> Tuple[List[List[float]], List[int]]:
        """
        Extract bounding boxes and cell IDs from instance mask.
        
        Args:
            mask: Instance segmentation mask
            apply_augment: Whether to apply box perturbation (training only)
        """
        boxes = []
        cell_ids = []
        
        img_h, img_w = mask.shape
        img_area = img_h * img_w
        
        # Relative size thresholds (for generalization across datasets)
        min_area_ratio = 0.0005  # 0.05% of image
        max_area_ratio = 0.15    # 15% of image
        min_box_area = img_area * min_area_ratio
        max_box_area = img_area * max_area_ratio
        
        for region in measure.regionprops(mask.astype(np.int32)):
            # Relative size filtering
            if region.area < min_box_area or region.area > max_box_area:
                continue
            
            y1, x1, y2, x2 = region.bbox
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(mask.shape[1], x2 + pad)
            y2 = min(mask.shape[0], y2 + pad)
            
            box = [x1, y1, x2, y2]
            
            # Apply perturbation during training
            if apply_augment and self.is_training:
                box = self._augment_box(box, mask.shape, noise_ratio=0.3)
            
            boxes.append(box)
            cell_ids.append(region.label)
        
        return boxes[:self.max_boxes], cell_ids[:self.max_boxes]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load data
        image = np.load(sample['image_path'])
        mask = np.load(sample['mask_path'])
        
        # Apply augmentation
        if self.transform is not None:
            # Albumentations expects uint8 or float32
            if image.max() > 1:
                image = image.astype(np.uint8)
            
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Manual resize if no albumentations
            if image.shape[:2] != self.target_size:
                image = transform.resize(image, self.target_size, preserve_range=True)
                mask = transform.resize(mask, self.target_size, order=0, preserve_range=True)
        
        # Normalize
        image = self._normalize_image(image)
        
        # Convert to RGB
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=0)
        elif image.ndim == 3 and image.shape[-1] == 3:
            image = image.transpose(2, 0, 1)
        else:
            image = np.stack([image, image, image], axis=0)
        
        # Generate boxes with cell IDs (apply augmentation during training)
        boxes, cell_ids = self._mask_to_boxes_with_ids(
            mask.astype(np.int32), 
            apply_augment=self.is_training
        )
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        cell_ids_tensor = torch.tensor(cell_ids, dtype=torch.long) if cell_ids else torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': torch.from_numpy(image).float(),
            'boxes': boxes_tensor,
            'cell_ids': cell_ids_tensor,  # Added: unique ID for each cell
            'mask': torch.from_numpy(mask.astype(np.int64)).long(),
            'sample_id': sample['sample_id'],
            'num_boxes': len(boxes)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function with cell_ids support."""
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    max_boxes = max(item['num_boxes'] for item in batch) if batch else 0
    max_boxes = max(max_boxes, 1)
    
    padded_boxes = []
    padded_cell_ids = []
    box_counts = []
    
    for item in batch:
        boxes = item['boxes']
        cell_ids = item['cell_ids']
        num_boxes = boxes.shape[0]
        box_counts.append(num_boxes)
        
        if num_boxes < max_boxes:
            box_padding = torch.zeros((max_boxes - num_boxes, 4))
            boxes = torch.cat([boxes, box_padding], dim=0)
            id_padding = torch.zeros((max_boxes - num_boxes,), dtype=torch.long)
            cell_ids = torch.cat([cell_ids, id_padding], dim=0)
        
        padded_boxes.append(boxes)
        padded_cell_ids.append(cell_ids)
    
    return {
        'image': images,
        'boxes': torch.stack(padded_boxes),
        'cell_ids': torch.stack(padded_cell_ids),  # Added
        'box_counts': torch.tensor(box_counts),
        'mask': masks,
        'sample_ids': [item['sample_id'] for item in batch]
    }


if __name__ == "__main__":
    # Test
    data_dir = "d:/AI/paper/CellSam/training_pairs_expanded"
    
    print("Testing augmented dataset...")
    dataset = AugmentedAllenDataset(data_dir, is_training=True)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample: {sample['sample_id']}")
        print(f"Image: {sample['image'].shape}")
        print(f"Mask: {sample['mask'].shape}")
        print(f"Boxes: {sample['boxes'].shape}")
        print(f"Cells: {sample['num_boxes']}")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"\nBatch test: images={batch['image'].shape}, masks={batch['mask'].shape}")
        print("✅ Augmented dataset working!")
