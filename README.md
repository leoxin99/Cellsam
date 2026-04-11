# CellSam Core Release for hiPSC-CM Whole-Cell Segmentation

This repository is a minimal, code-focused release of our core pipeline for
human hiPSC-derived cardiomyocyte whole-cell instance segmentation.

The public release is intentionally narrow. It keeps the components required to
reproduce the main project workflow while excluding paper-writing assets,
visualization scripts, temporary analysis code, and local experiment byproducts.

## What This Release Contains

The released core centers on four pieces:

1. `T28` three-channel decoder-only CellSAM adaptation for segmentation
2. `dapi_cm` biology-prior candidate generation from DAPI and Actn2
3. candidate-aware CellFinder fine-tuning for automatic box prompts
4. unified end-to-end inference and formal evaluation with a fixed protocol

## Upstream Basis and Attribution

This project builds on the public `cellSAM` codebase from
`vanvalenlab/cellSAM`.

- Vendored upstream source is included under `cellSAM_source/`
- The upstream license is preserved in `cellSAM_source/LICENSE.md`
- This release contains task-specific modifications for:
  - CellFinder adaptation on cardiomyocyte data
  - official-inspired unified inference
  - cardiomyocyte-specific candidate priors

## Repository Layout

```text
src/
  train.py
  augmented_dataset.py
  official_preprocess.py
  lora.py
  adapters/
  detection/
  inference/
  losses/
  metrics/
  config/

tools/
  train_cellfinder.py
  train_cellfinder_candidate_aware.py
  eval_h1b_e2e_formal_freeze.py

data/
  scripts/
  splits/

cellSAM_source/
```

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

This repository does not redistribute the Allen dataset itself.

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

## Main Training Commands

### 1. Train the segmentation mainline (`T28`)

```powershell
python src/train.py --config src/config/t28_planb_3ch.yaml --seed 42
```

Optional second seed:

```powershell
python src/train.py --config src/config/t28_planb_3ch.yaml --seed 123
```

### 2. Train the candidate-aware detector (`dapi_cm + CellFinder`)

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

The final inference pipeline released here is **not** the raw CellSAM inference
function used as-is.

Instead, this project uses a unified inference core that:

- adopts the official CellSAM preprocessing chain
- uses the project's locked segmentation path
- performs probability-based conflict resolution
- applies project-controlled clipping and postprocessing

The main entry point is:

- `src/inference/core.py`

## Scope of This Public Release

This repository intentionally does **not** include:

- raw or processed dataset files
- paper assets or thesis text
- temporary figures, logs, checkpoints, or exploratory tools
- local collaboration records and internal planning notes

## Suggested GitHub Description

If you want a short GitHub repository description, this would fit well:

`Minimal core release for human hiPSC-CM whole-cell segmentation with CellSAM adaptation and DAPI/Actn2-aware CellFinder prompting.`
