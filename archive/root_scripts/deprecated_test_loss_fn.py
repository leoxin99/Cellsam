# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: Superseded by unified inference core (Phase 0)
# Replacement entry points:
#   - Training:           src/train.py
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#   - Regression test:    tools/test_phase0_regression.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""Test the updated loss function."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'cellSAM_source'))
sys.path.insert(0, str(Path('.') / 'src'))

import torch
from train_expanded import CombinedLoss

# Test the updated loss function
loss_fn = CombinedLoss(pos_weight=10.0)
pred = torch.randn(1024, 1024)
target = torch.zeros(1024, 1024)
target[100:200, 100:200] = 1  # Small foreground region (~1% of image)

# Test without box
loss1 = loss_fn(pred, target)
print(f'Loss without box: {loss1.item():.4f}')

# Test with box
loss2 = loss_fn(pred, target, box=[90, 90, 210, 210])
print(f'Loss with box: {loss2.item():.4f}')

print('Loss function test passed!')
