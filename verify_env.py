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
