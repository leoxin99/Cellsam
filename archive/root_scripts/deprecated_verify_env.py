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
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

import segment_anything
print("segment-anything OK")

import skimage
print("scikit-image OK")

import sklearn
print("scikit-learn OK")

import albumentations
print("albumentations OK")

import dask
print("dask OK")

import tqdm
print("tqdm OK")

print("\n=== All packages verified successfully! ===")
