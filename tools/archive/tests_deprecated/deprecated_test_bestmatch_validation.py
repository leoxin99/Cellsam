"""
[DEPRECATED] Test Best-Match dice validation.

This script uses the pre-Phase-0 API:
  - Imports compute_best_match_dice (removed, now compute_bm_1to1_dice)
  - Expects validate() to return tuple (now returns dict)

Use tools/test_phase0_regression.py instead.
"""
import warnings
warnings.warn(
    "test_bestmatch_validation.py is deprecated. Use test_phase0_regression.py.",
    DeprecationWarning, stacklevel=2
)
import sys
sys.path.insert(0, 'cellSAM_source')
sys.path.insert(0, 'src')

# Early exit: this script uses removed APIs and will crash if run.
# - compute_best_match_dice -> now compute_bm_1to1_dice
# - validate() returned tuple -> now returns dict
if __name__ == "__main__":
    print("[DEPRECATED] This script uses removed APIs (compute_best_match_dice, validate tuple return).")
    print("Use instead: python tools/test_phase0_regression.py")
    sys.exit(1)

# Original code preserved below for reference (will not execute)
import torch
torch.cuda.empty_cache()

from train import load_config, validate  # compute_best_match_dice removed
from augmented_dataset import AugmentedAllenDataset, collate_fn, load_split_ids
from torch.utils.data import DataLoader
from cellSAM import get_model
