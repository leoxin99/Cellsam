# Chapter 4: Experimental Setup

## 4.1 Implementation Details

All experiments are conducted on the ALICE High-Performance Computing (HPC) cluster at Leiden University. Training is performed on NVIDIA L4 GPUs (24 GB VRAM) and NVIDIA A100 GPUs (80 GB VRAM). The codebase is implemented in Python using PyTorch, building upon the CellSAM source code.

The segmentation model follows the CellSAM architecture, which integrates a Vision Transformer (ViT-B) image encoder with the SAM prompt encoder and mask decoder. In our primary configuration (Best Config), the ViT-B encoder is frozen and only the prompt encoder (~6K parameters) and mask decoder (~4M parameters) are fine-tuned. All input images are resized to 1024×1024 pixels to match the SAM input specification.

Table 4.1 summarizes the training hyperparameters used across all experiments unless otherwise noted.

| Hyperparameter | Value |
|---|---|
| Model | CellSAM (ViT-B encoder + SAM mask decoder) |
| Frozen components | ViT-B image encoder, neck |
| Trainable components | Prompt encoder (~6K), mask decoder (~4M) |
| Image size | 1024 × 1024 |
| Batch size | 4 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| LR scheduler | Cosine annealing with linear warmup (5 epochs) |
| Epochs | 80 (with PQ-based early stopping, patience=15) |
| Training paradigm | Instance-level (one GT cell mask per bounding box) |
| Box prompt expansion | 10% of box width/height |
| Mixed precision | FP16 (torch.cuda.amp) |
| Data augmentation | RandomRotate90, HorizontalFlip, VerticalFlip, ShiftScaleRotate (shift=0.1, scale=0.1, rotate=30°), RandomBrightnessContrast |
| Post-processing | 6-step boundary smoothing → argmax probability assembly |

### Input Channel Configuration

In the default configuration (Best Config), only the brightfield (BF) channel is used as input. The single-channel grayscale image is replicated three times to match the three-channel input expected by the ViT-B encoder pretrained on RGB images. In the multi-channel experiments (§5.3), we additionally explore semantic channel mapping: R=BF, G=α-Actinin2, B=DAPI, using a lightweight 3×1 convolutional adapter layer to align the input distribution with the pretrained encoder.


## 4.2 Evaluation Protocol

We employ two evaluation settings to provide a comprehensive assessment:

**Oracle Evaluation.** Ground-truth bounding boxes are used as prompts. This isolates segmentation quality from detection errors and serves as the primary evaluation mode for ablation studies.

**End-to-End (E2E) Evaluation.** DAPI-detected bounding boxes are used as prompts, reflecting real-world deployment performance. The gap between Oracle and E2E results quantifies the impact of detection accuracy on overall segmentation.

All experiments are repeated with two random seeds (42 and 123) unless otherwise noted. We report the mean across seeds in the main tables and note standard deviations where relevant.

### Evaluation Metrics

We report the following metrics on the test set (73 images):

- **Panoptic Quality (PQ@0.5)**: PQ = SQ × RQ, where SQ (Segmentation Quality) is the mean IoU of matched instance pairs, and RQ (Recognition Quality) is the F1 score of detection at IoU ≥ 0.5. PQ is our primary metric as it jointly captures both segmentation and detection quality.

- **BM-1to1 Dice (Best-Match Dice)**: Hungarian-algorithm-based one-to-one matching between predicted and ground-truth instances, followed by computing the mean Dice coefficient per matched pair. This metric avoids double-counting and provides a reliable measure of per-instance segmentation accuracy.

- **Aggregated Jaccard Index (AJI)**: Measures the overall overlap between all predicted and ground-truth instances with a global penalty for false positives and false negatives.


## 4.3 Baseline Methods

We compare our approach against the following methods, all evaluated on the same test set of 73 images using Oracle bounding boxes where applicable:

- **CellSAM (Original)**: The pretrained CellSAM checkpoint without any fine-tuning, using our GT bounding boxes as prompts. This represents the zero-shot performance of CellSAM on cardiomyocytes.

- **SAM ViT-B**: The original Segment Anything Model (ViT-B variant) without cell-specific training, using GT bounding boxes. This serves as a non-domain-adapted baseline.

- **MedSAM**: A SAM model fine-tuned on a large-scale medical image dataset (~1.5M image-mask pairs across 11 modalities). MedSAM represents the upper-bound reference for SAM-based approaches, as it benefits from extensive medical domain pretraining. We evaluate it with GT bounding boxes.

- **Cellpose**: A widely-used deep learning tool for cell segmentation that uses gradient flow representations. We evaluate it with default settings on brightfield images.

- **StarDist**: A cell segmentation method based on star-convex polygon representations, designed primarily for convex-shaped nuclei. We evaluate it with default pretrained models.

- **SAMCell**: A SAM-based cell segmentation pipeline. We evaluate it with the provided pretrained weights.

All baseline evaluations use the unified evaluation framework (`inference/core.py` + `tools/baseline_eval.py`) to ensure consistent metric computation.
