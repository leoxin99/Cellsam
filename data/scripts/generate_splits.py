"""
Generate splits from PROCESSED files (not raw TIFF).
This ensures sample IDs match the actual files in data/processed.
"""
from pathlib import Path
import random

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "images"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

def main():
    # Get all sample IDs from processed directory
    all_npy = sorted(PROCESSED_DIR.glob("*.npy"))
    all_ids = [f.stem for f in all_npy]
    
    print(f"Total processed samples: {len(all_ids)}")
    
    # Shuffle with fixed seed
    random.shuffle(all_ids)
    
    # Split
    n_total = len(all_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train + n_val]
    test_ids = all_ids[n_train + n_val:]
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Save
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(SPLITS_DIR / "train_ids.txt", "w") as f:
        f.write("\n".join(train_ids))
    
    with open(SPLITS_DIR / "val_ids.txt", "w") as f:
        f.write("\n".join(val_ids))
    
    with open(SPLITS_DIR / "test_ids.txt", "w") as f:
        f.write("\n".join(test_ids))
    
    print(f"\nSplit files saved to: {SPLITS_DIR}")
    print(f"Sample ID format: {train_ids[0]}")

if __name__ == "__main__":
    main()
