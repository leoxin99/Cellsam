# CellSAM SAM Branch Audit (2026-02-21)

## 1. Scope

This note audits two questions:

1. What is the **loss design of CellSAM's SAM branch** (not CellFinder)?
2. What is the end-to-end CellSAM structure and how does it map to code?

Sources used:

- CellSAM preprint (bioRxiv v3 full text):  
  `https://www.biorxiv.org/content/10.1101/2023.11.17.567630v3.full`
- CellSAM Nature Methods paper page:  
  `https://www.nature.com/articles/s41592-024-02499-w`
- CellSAM PMC page:  
  `https://pmc.ncbi.nlm.nih.gov/articles/PMC11223854/`
- Local code snapshot: `cellSAM_source/`

---

## 2. SAM Branch Loss: Evidence-Based Conclusion

## 2.1 What can be confirmed with high confidence

1. CellSAM is trained in a two-stage fashion:
- Stage 1 trains a detector (CellFinder / AnchorDETR style).
- Stage 2 fine-tunes the SAM-related segmentation pathway with box prompts and masks.

2. The public `cellSAM_source` code snapshot in this repo is primarily inference/evaluation-oriented:
- `cellSAM_source/README.md:8` explicitly describes inference code.
- `cellSAM_source/cellSAM/sam_inference.py` has prediction flow but no training loop (`optimizer`, `backward`, `criterion`).

## 2.2 What cannot be claimed as code-proven from current public snapshot

1. Exact stage-2 SAM loss formula and weight coefficients are not directly recoverable from this local public snapshot, because the stage-2 training script is not present.
2. Therefore, statements like "exactly BCE with fixed coefficient X.Y" should be treated as **paper-text-level claim**, not "code-reproduced claim" in this repo.

## 2.3 Practical takeaway for this project

When writing paper/report:

- You can state: "CellSAM uses a two-stage training strategy and our work modifies downstream fine-tuning objective design."
- You should avoid over-claiming unrecoverable implementation details from `cellSAM_source`.
- If exact loss terms are required for publication rigor, retrieve supplementary training code/materials from authors or official reproduction assets beyond this snapshot.

---

## 3. CellSAM Architecture and Principle (with Code Mapping)

## 3.1 High-level flow

1. Load model weights and config.
2. If no boxes are provided, CellFinder proposes candidate boxes.
3. For each box, run SAM prompt encoder + mask decoder to get one mask.
4. Merge instance masks into a final instance map.
5. For WSI/large images, run tiled segmentation and link labels across tile overlaps.

## 3.2 Code mapping

1. Model API entry:
- `cellSAM_source/cellSAM/model.py:50` `get_model`
- `cellSAM_source/cellSAM/model.py:114` `segment_cellular_image`

2. Pipeline entry:
- `cellSAM_source/cellSAM/cellsam_pipeline.py:54` `cellsam_pipeline`

3. Core model class:
- `cellSAM_source/cellSAM/sam_inference.py:123` `class CellSAM`

4. CellFinder (box proposal):
- `cellSAM_source/cellSAM/sam_inference.py:76` `class CellfinderAnchorDetr`
- Dynamic thresholding logic (KMeans-assisted):  
  `cellSAM_source/cellSAM/sam_inference.py:247`, `cellSAM_source/cellSAM/sam_inference.py:253`

5. SAM box-prompt segmentation:
- Prompt encoder call: `cellSAM_source/cellSAM/sam_inference.py:333`
- Mask decoder call: `cellSAM_source/cellSAM/sam_inference.py:339`
- IoU screening: `cellSAM_source/cellSAM/sam_inference.py:350`
- Mask resize/postprocess: `cellSAM_source/cellSAM/sam_inference.py:354`

6. Multi-instance conflict resolution:
- Per-instance IDs stacked then merged using `np.max`:  
  `cellSAM_source/cellSAM/sam_inference.py:388`, `cellSAM_source/cellSAM/sam_inference.py:391`
- This is a rule-based assignment (ID-order aggregation), not a global optimization solver.

7. WSI tiling + cross-tile relabel:
- `cellSAM_source/cellSAM/wsi.py:38` `segment_wsi`
- `cellSAM_source/cellSAM/wsi.py:138` `link_labels`
- `cellSAM_source/cellSAM/wsi.py:149` `label_adjacency_graph`

---

## 4. Separation from CellFinder Loss

CellFinder-related losses visible in code are mainly in AnchorDETR modules:

- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:163` classification loss path
- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:208` box loss path
- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:234` optional mask loss path
- `cellSAM_source/cellSAM/AnchorDETR/models/matcher.py:101` Hungarian cost composition

These are detector-side mechanics and should not be conflated with the stage-2 SAM branch training objective.

---

## 5. Comparison to Our Current Project Loss Stack

Our project loss is explicitly richer and task-specialized:

- `src/losses/combined.py:29` BoundaryLoss
- `src/losses/combined.py:105` AJILoss
- `src/losses/combined.py:167` TopologyLoss
- `src/losses/combined.py:225` SizeLoss
- `src/losses/combined.py:290` ContourLoss
- `src/losses/combined.py:381` NeighborIntrusionLoss
- `src/losses/combined.py:414` OverlapMutexLoss
- `src/losses/combined.py:439` CombinedLoss orchestration

Meaning: our training objective goes beyond generic mask fitting and explicitly models boundary quality, instance behavior, and neighbor overlap penalties for cardiomyocyte adhesion scenarios.

---

## 6. Recommended Citation-Safe Wording

Use wording like:

- "Based on the publicly available CellSAM inference repository and paper text, we verified the two-stage framework and prompt-based segmentation mechanism. Exact stage-2 training-loss implementation details are not fully recoverable from the public inference snapshot, so we report only evidence-backed claims."

