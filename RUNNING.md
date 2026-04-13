# Running CellSam

This document contains the practical steps for preparing data, training the
core models, and running the formal end-to-end evaluation.

## Environment

We recommend Python 3.10+ and PyTorch.

A practical environment should include at least:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `scikit-image`
- `opencv-python`
- `tifffile`
- `pyyaml`
- `tqdm`
- `albumentations`
- `pycocotools`
- `boto3`
- `requests`
- `segment-anything`

Depending on your local setup, the vendored `cellSAM_source` may require a few
additional upstream packages.

## Data Preparation

The intended workflow is:

1. download the raw TIFF files from the Allen public source
2. convert them into the project `data/processed` format
3. use the fixed split files in `data/splits/`

### Step 1: Download raw data

```powershell
python data/scripts/download_full_segmented.py
```

This downloads raw TIFF annotations into:

```text
data/raw/allen_segmented_fields_full/
```

### Step 2: Build processed training pairs

```powershell
python data/scripts/extract_expanded_pairs.py
```

This produces:

```text
data/processed/images/*.npy
data/processed/masks/*.npy
```

Each processed image is stored as:

```text
[BF, DAPI, Actn2]
```

### Step 3: Use the fixed split protocol

The fixed split files are included:

- `data/splits/train_ids.txt`
- `data/splits/val_ids.txt`
- `data/splits/test_ids.txt`

If you need to regenerate them from processed files:

```powershell
python data/scripts/generate_splits.py
```

## Training

### 1. Train the segmentation mainline

```powershell
python src/train.py --config src/config/t28_planb_3ch.yaml --seed 42
```

Optional second seed:

```powershell
python src/train.py --config src/config/t28_planb_3ch.yaml --seed 123
```

### 2. Train the candidate-aware detector

Example command for the `dapi_cm` biology-prior route:

```powershell
python tools/train_cellfinder_candidate_aware.py `
  --seed 42 `
  --num-queries 35 `
  --candidate-mode dapi_cm `
  --profile-name locked_eval `
  --prior-mode strict `
  --query-output-mode candidate_aligned
```

Optional second seed:

```powershell
python tools/train_cellfinder_candidate_aware.py `
  --seed 123 `
  --num-queries 35 `
  --candidate-mode dapi_cm `
  --profile-name locked_eval `
  --prior-mode strict `
  --query-output-mode candidate_aligned
```

### 3. Train the baseline fine-tuned CellFinder detector

```powershell
python tools/train_cellfinder.py --seed 42 --num-queries 50
```

## Formal End-to-End Evaluation

After checkpoints are available under `checkpoints/`, run:

```powershell
python tools/eval_h1b_e2e_formal_freeze.py
```

By default this evaluates the fixed detector-to-segmenter protocol and writes:

```text
results/h1b_e2e_formal_t28_single_source.json
results/h1b_e2e_formal_t28_single_source.md
```

## Important Implementation Boundary

The final inference pipeline released here is not the raw CellSAM inference
function used as-is.

Instead, this project uses a unified inference core that:

- adopts the official CellSAM preprocessing chain
- uses the project's locked segmentation path
- performs probability-based conflict resolution
- applies project-controlled clipping and postprocessing

The main entry point is:

- `src/inference/core.py`

## Reproducibility Note

The formal detector-to-segmenter evaluation keeps different query counts for
different detector arms:

- the baseline fine-tuned CellFinder line uses `50`
- the candidate-aware detector lines use `35`

That difference is intentional and is part of the locked evaluation protocol in
`tools/eval_h1b_e2e_formal_freeze.py`.
