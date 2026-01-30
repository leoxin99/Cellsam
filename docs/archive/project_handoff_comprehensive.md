# CellSAM Project Comprehensive Handoff

> **Date**: 2026-01-27
> **From**: Antigravity (Agent)
> **To**: Claude (Incoming Agent)
> **Status**: Detection Pipeline Optimized, Ready for Model Architecture Changes

---

## 1. Project Overview (High Level)

**Goal**: Adapt Segment Anything (SAM) for **high-precision cardiomyocyte segmentation** using single-cell crops.
**Key Challenge**: Cardiomyocytes are large, elongated, and often binucleated. Standard SAM (trained on natural images) struggles with:
1.  **Binucleation**: Treating one cell as two.
2.  **Boundary**: Missing Z-discs at the cell ends.
3.  **Ambiguity**: Confusing adjacent touching cells.

**Current Architecture**:
-   **Input**: 3 Channels (BF, Actn2, DAPI), currently utilizing DAPI+Actn2 for prompting.
-   **Model**: SAM (ViT-H) with frozen Image Encoder.
-   **Strategy**: "Detection-then-Segmentation". We detect the cell first to generate a precise Bounding Box Prompt, then feed it to SAM.

---

## 2. Current Status: Detection Pipeline (COMPLETED ✅)

We have just finished optimizing the **Detection & Prompting** module.

### The "Hybrid DAPI+Actn2" Solution (`src/detection/dapi.py`):
1.  **Nucleus Detection**:
    -   Algorithm: Otsu thresholding + `min_area=3000` filter.
    -   **Edge Handling**: `margin=50px` (Excludes cells touching image edges).
2.  **Binucleation Handling**:
    -   Logic: Merge two nuclei if distance < **1.2 * avg_diameter** (Dynamic threshold).
    -   Why 1.2? To avoid merging neighbors in dense clusters.
3.  **Adaptive Bounding Box**:
    -   Problem: Fixed padding is bad for irregular cells.
    -   Solution: Detect Actn2 Z-lines around the nucleus. Use the Z-line cloud to define the Box extent.
    -   Result: Box "hugs" the cell shape.

**Files**:
-   `src/detection/dapi.py`: Core logic.
-   `tools/visualize_detection_comparison.py`: Verification script.

---

## 3. Next Major Task: 3-Channel SAM Adaptation (The "Act2n/Claude" Plan) 🚀

**This is the immediate goal for the new agent.**

**Objective**: Implement the **"Semantic Channel Mapping"** and **"Channel Adapter"** design previously analyzed in `claude_pipeline_analysis.md`.

### The Plan (Detailed in `claude_pipeline_analysis.md`):

1.  **Preprocessing: Semantic Channel Mapper**
    *   Do NOT just stack raw channels. Map biologically distinct channels to RGB semantics:
    *   **R (Channel 0) ← α-actinin (Actn2)**: Structure. Preprocess: Percentile truncation (P0.5-P99.5).
    *   **G (Channel 1) ← Phase/BF**: Context. Preprocess: CLAHE enhancement.
    *   **B (Channel 2) ← DAPI**: Localization. Preprocess: Gaussian smooth (sigma=3).
    *   *Result*: A pseudo-RGB image optimized for features SAM understands.

2.  **Model Architecture: Channel Adapter**
    *   Instead of just modifying the input layer, implement a **learnable adapter**.
    *   **Recommended**: `IndependentChannelAdapter` (Simulates IC-ViT).
    *   **Mechanism**: Independent convolutions per channel to align feature distributions before entering the frozen ViT.

3.  **Integration Steps**:
    *   Locate the specific "Claude Pipeline" code (referenced in the analysis artifact).
    *   Merge `SemanticChannelMapper` into `src/data/`.
    *   Integrate `IndependentChannelAdapter` into the model definition in `src/train.py`.

### Action Items for Claude:
1.  **Retrieve Design**: Read `claude_pipeline_analysis.md` (Artifact) carefully.
2.  **Implement Mapper**: Create `src/data/preprocessing.py` implementing the R/G/B mapping.
3.  **Implement Adapter**: Add the adapter module to the image encoder input.

---

## 4. System Context for Claude

Use this prompt to initialize the session:

```markdown
# Role
You are the Lead AI Researcher for the CellSAM project. You are taking over a validated codebase.

# Project State
- **Repo**: `d:/AI/paper/CellSam`
- **Validated Detection**: `src/detection/dapi.py` (Hybrid DAPI+Actn2 box).
- **Design Reference**: `claude_pipeline_analysis.md` (Artifact).

# Your Mission
**Implement the "Semantic Channel Mapping" & "Channel Adapter" for SAM.**
1.  **Refactor Data Loading**: Implement the R=Actn2, G=Phase, B=DAPI mapping strategy.
2.  **Model Adaptation**: Implement the `IndependentChannelAdapter` to feed these channels into the frozen ViT.
3.  **Train**: Fine-tune the Mask Decoder with this new input pipeline.
```

---

## 5. FAQ (User Questions Answered)

**Q: Where are the statistics run?**
A: **The "Dev Set" (Subset of Full Data)**.
-   The dataset used in `analyze_stats_final.py` is the **first 50 images** (`allen_segmented_fields_full` sliced `[:50]`).
-   This corresponds to the defined **Dev Set** (see `docs/dataset_parameters.md`).
-   *Correction*: It is NOT the full 478-image dataset (Test set must remain unseen), but it IS the "Full set of designated development data".

**Q: Why 1.2x for binucleation?**
A: 1.5x proved too aggressive, merging neighbors. 1.2x is conservative, ensuring we only merge nuclei that are clearly part of the same cytoplasmic mass.

**Q: Where is the 3-channel design record?**
A: **`claude_pipeline_analysis.md`**. This artifact details the precise mapping (R=Actn2, G=Phase, B=DAPI) and adapter strategy.
