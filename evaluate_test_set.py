"""
Evaluate trained model on independent test set.
Creates test set from training samples and evaluates performance.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset

# Configuration
CHECKPOINT_DIR = "d:/AI/paper/CellSam/checkpoints/expanded_20260108_034352"
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pt"
FULL_CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/checkpoint_epoch14.pt"  # Contains train/val split
DATA_DIR = "d:/AI/paper/CellSam/data/processed"

def calculate_dice(pred, gt):
    """Calculate Dice score."""
    intersection = (pred * gt).sum()
    return (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)


def evaluate_on_dataset(model, dataset, device):
    """Evaluate model on dataset with per-cell Dice calculation."""
    model.eval()

    dice_scores = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[idx]

            image = sample['image'].unsqueeze(0).to(device)
            boxes = sample['boxes'].to(device)
            cell_ids = sample['cell_ids'].to(device)
            gt_mask = sample['mask'].to(device)
            num_boxes = sample['num_boxes']

            if num_boxes == 0:
                continue

            # Encode image
            img_preprocessed = model.sam_preprocess(image)
            embedding = model.model.image_encoder(img_preprocessed)

            # Evaluate each cell
            for box_idx in range(min(num_boxes, 20)):
                box = boxes[box_idx:box_idx+1].unsqueeze(0)
                cell_id = cell_ids[box_idx].item()

                # Predict mask
                sparse_emb, dense_emb = model.model.prompt_encoder(
                    points=None, boxes=box, masks=None
                )

                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )

                pred_mask = F.interpolate(
                    low_res_masks, size=(gt_mask.shape[0], gt_mask.shape[1]),
                    mode='bilinear', align_corners=False
                ).squeeze()

                # Get GT for this cell
                gt_cell_mask = (gt_mask == cell_id).float()

                # Calculate Dice
                pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
                dice = calculate_dice(pred_binary, gt_cell_mask)
                dice_scores.append(dice.item())

    return dice_scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load checkpoint to get data split
    print("Loading checkpoint for data split...")
    full_checkpoint = torch.load(FULL_CHECKPOINT_PATH, map_location=device, weights_only=False)

    train_ids = full_checkpoint.get('train_ids', [])
    val_ids = full_checkpoint.get('val_ids', [])

    print(f"Original split:")
    print(f"  Training: {len(train_ids)} samples")
    print(f"  Validation: {len(val_ids)} samples")

    # Create test set from training samples (randomly select 10)
    np.random.seed(42)
    test_ids = np.random.choice(train_ids, size=10, replace=False).tolist()

    print(f"\nTest set (10 samples from training):")
    for i, test_id in enumerate(test_ids, 1):
        print(f"  {i}. {test_id[:40]}...")

    # Load model
    print("\nLoading best model weights...")
    model = get_model()
    best_weights = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_weights)
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = AugmentedAllenDataset(
        data_dir=DATA_DIR,
        is_training=False,  # No augmentation
        sample_ids=test_ids
    )

    # Also evaluate on validation set for comparison
    val_dataset = AugmentedAllenDataset(
        data_dir=DATA_DIR,
        is_training=False,
        sample_ids=val_ids
    )

    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    test_dice_scores = evaluate_on_dataset(model, test_dataset, device)

    # Evaluate on validation set
    print("\n" + "="*60)
    print("EVALUATING ON VALIDATION SET")
    print("="*60)
    val_dice_scores = evaluate_on_dataset(model, val_dataset, device)

    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\nTest Set ({len(test_dice_scores)} cells):")
    print(f"  Mean Dice: {np.mean(test_dice_scores):.4f}")
    print(f"  Std Dice:  {np.std(test_dice_scores):.4f}")
    print(f"  Min Dice:  {np.min(test_dice_scores):.4f}")
    print(f"  Max Dice:  {np.max(test_dice_scores):.4f}")
    print(f"  Median:    {np.median(test_dice_scores):.4f}")

    print(f"\nValidation Set ({len(val_dice_scores)} cells):")
    print(f"  Mean Dice: {np.mean(val_dice_scores):.4f}")
    print(f"  Std Dice:  {np.std(val_dice_scores):.4f}")
    print(f"  Min Dice:  {np.min(val_dice_scores):.4f}")
    print(f"  Max Dice:  {np.max(val_dice_scores):.4f}")
    print(f"  Median:    {np.median(val_dice_scores):.4f}")

    # Dice distribution
    print("\nDice Score Distribution (Test Set):")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins)-1):
        count = sum(bins[i] <= d < bins[i+1] for d in test_dice_scores)
        pct = count / len(test_dice_scores) * 100
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {count:3d} cells ({pct:5.1f}%)")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
