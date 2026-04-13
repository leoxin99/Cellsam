# CellSam

CellSam is our minimal public code release for human hiPSC-derived
cardiomyocyte whole-cell instance segmentation.

This repository keeps the core training, detection, inference, and evaluation
code used in the project while excluding paper assets, temporary analysis code,
local experiment byproducts, and dataset files.

## Overview

The released core centers on four pieces:

1. three-channel decoder-only CellSAM adaptation for segmentation
2. DAPI/Actn2-aware candidate generation for cardiomyocyte localization
3. candidate-aware CellFinder fine-tuning for automatic box prompting
4. unified end-to-end inference and fixed-protocol evaluation

## Repository Layout

```text
src/
  core training, preprocessing, inference, losses, metrics, configs

tools/
  detector training and end-to-end evaluation entrypoints

data/
  dataset preparation scripts and fixed split files

cellSAM_source/
  vendored upstream cellSAM source with task-specific modifications
```

## Data and Running Instructions

This repository does not redistribute the Allen dataset itself.

For environment setup, data preparation, training commands, and evaluation
steps, see:

- `RUNNING.md`

## Upstream Basis and Attribution

This project builds on the public `cellSAM` codebase from
`vanvalenlab/cellSAM`.

- Vendored upstream source is included under `cellSAM_source/`
- The upstream license is preserved in `cellSAM_source/LICENSE.md`
- This release contains task-specific modifications for:
  - CellFinder adaptation on cardiomyocyte data
  - unified official-inspired inference
  - cardiomyocyte-specific candidate priors

## Scope

This public release intentionally does not include:

- raw or processed dataset files
- paper or thesis assets
- temporary figures, logs, checkpoints, or exploratory tools
- local collaboration records and internal planning notes
