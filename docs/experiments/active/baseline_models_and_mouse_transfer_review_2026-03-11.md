# Baseline Models Review + Human-to-Mouse Cardiomyocyte Transfer Assessment

> Author: A1 (Codex)
> Date: 2026-03-11
> Purpose: consolidate the major baseline models used or referenced in this project, explain their design/role relative to our cardiomyocyte task, and assess whether the current Allen hiPSC-CM pipeline can transfer to mouse cardiomyocyte segmentation.
> Status: research review / experiment-planning document; not a completed training run.

## 1. Executive Summary

### 1.1 What this document answers

1. What are the major baseline models in the current thesis narrative, and what question does each baseline answer?
2. Why is the current project scientifically valuable even when compared against strong general or medical foundation baselines?
3. Our dataset is human hiPSC-derived cardiomyocytes. Can the current pipeline be reused for mouse cardiomyocytes, and if so, what must be changed?

### 1.2 Main conclusions

1. The current locked thesis baselines are methodologically sufficient: `Cellpose cyto3`, `SAM ViT-B`, `CellSAM pretrained`, and `MedSAM` cover four distinct comparison questions:
   - generic cell segmentation transfer,
   - generic promptable segmentation without cell adaptation,
   - cell-domain foundation transfer,
   - large-scale medical-domain promptable transfer.
2. On our locked `test73` table, `T27a` outperforms all of them, including `MedSAM`.
3. This is not just a performance claim. The project contributes in three layers:
   - a code-level audit of the released CellSAM artifact,
   - a cardiomyocyte-specific fine-tuning route,
   - a split-aware evaluation discipline that separates stable evidence from audit-only evidence.
4. The current dataset is **human hiPSC-cardiomyocytes**, not mouse cardiomyocytes.
5. Transfer to mouse cardiomyocytes is plausible, but zero-shot reuse should not be assumed. The amount of modification depends on the mouse imaging scenario:
   - cultured mouse CM with `DAPI + structural marker + BF/phase`: moderate adaptation,
   - isolated adult rod-shaped mouse CM: stronger adaptation,
   - tissue sections / in vivo histology: major redesign.

## 2. Dataset Scope of the Current Project

### 2.1 Current project data identity

The current project dataset is the Allen Institute hiPSC-derived cardiomyocyte dataset:
- `478` total images,
- split `334 / 71 / 73`,
- multi-channel microscopy,
- task = whole-cell instance segmentation of hiPSC-derived cardiomyocytes.

Local project evidence:
- `docs/report_2.19.md`
- `docs/paper_writing/paper_preparation.md`
- `docs/paper_writing/chapters/ch3_dataset_and_evaluation.md`

This matters because all thesis claims about `T27a` are currently **human hiPSC-CM claims**, not generic mammalian cardiomyocyte claims.

## 3. Baseline Model Review

### 3.1 Why these baselines matter

The current thesis should not describe all baselines as doing the same job. They answer different questions:

| Baseline | What it tests | Why it matters for the thesis |
|---|---|---|
| `Cellpose cyto3` | Can a strong generalist cell segmenter transfer directly? | Tests whether foundation-model adaptation is even needed |
| `SAM ViT-B` | Can generic promptable segmentation work with GT boxes but no cell-domain adaptation? | Tests the value of cell-domain pretraining/fine-tuning |
| `CellSAM pretrained` | Does a released cell-domain foundation model already solve cardiomyocytes? | Tests zero-shot cell-domain transfer |
| `MedSAM` | Does large-scale medical pretraining outperform cell-domain adaptation? | Tests whether our gain is only due to foundation scale or truly task-specific adaptation |
| `T27a` | Can an audited, cardiomyocyte-specific CellSAM adaptation beat all of the above? | This is the project's central claim |

### 3.2 Current locked results on our task

Current thesis-aligned `test73` table (from local paper docs):

| Model | PQ | BM-Dice | AJI | Notes |
|---|---:|---:|---:|---|
| Cellpose v3.1.1 (`cyto3`, `d=250`) | 0.273 | 0.505 | 0.285 | `T31`, paper-aligned baseline |
| SAM ViT-B | 0.286 | 0.631 | 0.440 | no cell-specific training |
| CellSAM pretrained | 0.434 | 0.682 | 0.499 | corrected official path |
| MedSAM | 0.576 | 0.771 | 0.634 | strong external baseline |
| **T27a Plan B Decoder-Only** | **0.659** | **0.800** | **0.669** | current locked mainline |

Local evidence:
- `docs/paper_writing/paper_preparation.md`
- `docs/paper_writing/chapters/ch5_results.md`

### 3.3 SAM ViT-B

**Paper / source**:
- Kirillov et al., Segment Anything, ICCV 2023
- URL: `https://arxiv.org/abs/2304.02643`

**Core design**:
- promptable segmentation foundation model,
- image encoder (`ViT`), prompt encoder, mask decoder, IoU head,
- designed to segment arbitrary objects given prompts (points / boxes / masks).

**What it contributes as a baseline here**:
- It tells us whether generic promptable segmentation plus GT boxes is already sufficient.
- It is the cleanest test of "does cell-domain specialization matter at all?"

**Why it underperforms on our task**:
- no cell-domain pretraining,
- no cardiomyocyte-specific fine-tuning,
- no adaptation to elongated adherent whole-cell morphology.

**Thesis value against SAM ViT-B**:
- If `T27a > SAM ViT-B`, the gain cannot be attributed only to box prompting or generic SAM priors.
- It supports the claim that cardiomyocyte-specific adaptation is necessary.

### 3.4 CellSAM pretrained

**Paper / source**:
- Israel et al., CellSAM, Nature Methods
- DOI used in local docs: `https://doi.org/10.1038/s41592-025-02879-w`
- official repo: `https://github.com/vanvalenlab/cellSAM`

**Core design**:
- CellSAM is not just SAM inference on cell images.
- The paper describes a two-stage strategy:
  1. Stage 1 trains `CellFinder + ViT backbone` for detection.
  2. Stage 2 fine-tunes the segmentation-side alignment (paper says neck-only; public training script not released in a line-by-line reproducible form).
- The released artifact contains multiple branches / weights, which is why our project had to audit `model` vs `model_cp` and the official inference path.

**What it contributes as a baseline here**:
- It answers: "If we take a released cell-domain foundation model as-is, how far does it transfer to human cardiomyocytes?"

**Why it is still insufficient for our thesis task**:
- cardiomyocyte morphology is underrepresented relative to the broad cell benchmarks CellSAM was designed for,
- released path details needed auditing before fair use,
- detector quality remains a separate bottleneck in E2E use.

**Thesis value against CellSAM pretrained**:
- Our project is not merely training "another SAM".
- It is showing that a released cell foundation model still needs **task-specific adaptation + path auditing** to become a strong cardiomyocyte system.

### 3.5 MedSAM

**Paper / source**:
- MedSAM, Nature Communications 2024
- repo: `https://github.com/bowang-lab/MedSAM`

**Core design**:
- promptable medical adaptation of SAM,
- trained on large-scale medical image-mask pairs (local project notes record `1.57M` pairs; see `docs/technical/update_cellsam.md`).
- designed to transfer across medical imaging modalities rather than cell microscopy alone.

**What it contributes as a baseline here**:
- It is the strongest "large-scale medical pretraining" comparison.
- It asks whether massive generic medical pretraining beats domain-specific cardiomyocyte tuning.

**Why it is strong but still not sufficient on our task**:
- It has broad medical-domain priors,
- but it is not optimized for adherent cultured cardiomyocyte whole-cell morphology,
- and it does not incorporate our project's path audit / cardiomyocyte-specific decoder tuning.

**Thesis value against MedSAM**:
- This is the strongest external baseline in the current table.
- If `T27a > MedSAM`, the thesis can claim more than "better than generic cell tools".
- It can claim that **cardiomyocyte-specific adaptation plus audited model-path control can outperform even a strong medical foundation baseline on this task**.

### 3.6 Cellpose (`cyto3`)

**Paper / source**:
- original Cellpose paper: Stringer et al., Nature Methods 2021
- official docs: `https://cellpose.readthedocs.io/`
- local model-zoo review: `docs/technical/cellpose_builtin_models_reference.md`

**Core design**:
- prompt-free cell segmentation,
- uses a learned flow-field / dynamics formulation to separate instances,
- `cyto3` is the current generalist whole-cell pretrained model used in paper-aligned evaluation.

**What it contributes as a baseline here**:
- It answers whether a mature, strong generalist cell segmentation system can solve the task without box prompts and without cardiomyocyte-specific tuning.

**Why it underperforms on our task**:
- `cyto3` is a strong generalist, but not cardiomyocyte-specific,
- whole-cell separation in adherent elongated hiPSC-CM scenes is a hard out-of-domain case,
- current paper-aligned rerun still remains weak relative to foundation-model routes on this dataset.

**Thesis value against Cellpose**:
- This comparison justifies why the project is not solving a trivial problem already handled by an off-the-shelf cell segmenter.
- It supports the argument that cardiomyocyte whole-cell segmentation is structurally harder than standard whole-cell benchmarks.

### 3.7 Optional contextual baselines (do not need to be central in the main table)

These are useful for related-work framing but do not need to dominate the thesis result table:
- `StarDist`: star-convex instance modeling; strong classical microscopy baseline, but shape assumptions are strained by irregular cardiomyocyte spread.
- `U-Net / DeepLabV3+`: strong task-specific segmentation architectures; useful historical context, but less central than the currently audited baselines above.

## 4. Stronger Thesis Narrative: Why Our Project Is Valuable

### 4.1 The thesis should not make a single weak claim

A weak claim would be:
- "We fine-tuned CellSAM and got a better number."

A stronger, defensible claim is:
- We started from a released cell foundation model whose public artifact contains non-trivial path ambiguity.
- We audited the actual segmentation branch and inference path.
- We then designed a cardiomyocyte-specific fine-tuning route that is evaluated with split-aware discipline.
- The resulting system beats generic segmentation transfer (`Cellpose`, `SAM ViT-B`), cell-domain zero-shot transfer (`CellSAM pretrained`), and a strong medical foundation baseline (`MedSAM`) on the locked human hiPSC-CM test set.

### 4.2 What is scientifically stronger than just "better PQ"

The project's value is strongest when written as four layers:

1. **Artifact audit value**
   - CellSAM public weights and paths are not trivial to interpret.
   - Our project contributes a code-level clarification of what is actually being used.

2. **Method value**
   - `T27a` is not arbitrary tuning; it is a domain-specific decoder adaptation built on an audited path.

3. **Evaluation value**
   - The project separates Oracle segmentation, E2E prompting, and path-audit evidence.
   - This prevents invalid mixing of unstable or split-mismatched claims.

4. **Task value**
   - hiPSC-cardiomyocytes are unusually hard segmentation targets: elongated, adherent, anisotropic, and weak-boundary.
   - Beating strong baselines here is more meaningful than beating them on compact nucleus-like objects.

## 5. Human vs Mouse Cardiomyocytes: What Changes

### 5.1 Important biological / imaging differences

The current dataset is human **hiPSC-derived** cardiomyocytes, not adult human myocardium and not mouse cardiomyocytes.

The most relevant differences for segmentation transfer are:

| Dimension | Current project: human hiPSC-CM | Typical mouse cardiomyocyte scenario | Expected segmentation impact |
|---|---|---|---|
| Maturation | immature / in vitro derived | often adult or neonatal primary mouse CM | texture and boundary statistics shift |
| Shape | spread, variable, anisotropic, often irregular | adult mouse CM often more rod-shaped and organized | shape prior changes |
| Nucleation | often mono-nuclear or mixed immature states | adult mouse ventricular CM are frequently binucleated | nucleus-based box logic must be retuned |
| Sarcomere organization | less mature / heterogeneous | adult mouse CM often stronger sarcomeric structure | Actn2/Z-line cues may behave differently |
| Imaging context | Allen multi-channel cultured field images | could be isolated cells, culture, or tissue sections | pipeline transfer depends strongly on imaging setup |

External review support:
- mouse cardiomyocytes are commonly described as rod-shaped and largely binucleated in adulthood,
- hiPSC-CM are widely described as immature relative to adult cardiomyocytes,
- human vs rodent cardiomyocytes differ in maturation state, nucleation, electrophysiology, and structural organization.

Representative sources:
- `https://pmc.ncbi.nlm.nih.gov/articles/PMC8718008/`
- `https://pmc.ncbi.nlm.nih.gov/articles/PMC6390119/`
- `https://pmc.ncbi.nlm.nih.gov/articles/PMC8867505/`
- `https://pmc.ncbi.nlm.nih.gov/articles/PMC8034626/`

### 5.2 Can the current project be used on mouse cardiomyocytes?

**Yes, but not as a claim of direct zero-shot equivalence.**

The practical answer depends on the mouse dataset type.

#### Case A. Mouse cultured cardiomyocytes with similar channels
Example:
- `DAPI + alpha-actinin + BF/phase`
- single-cell or sparse monolayer culture

This is the easiest transfer case.

What can be reused:
- audited CellSAM segmentation branch,
- `T27a` training/inference code structure,
- DAPI-based box generation framework,
- most evaluation metrics.

What must change:
1. regenerate dataset statistics,
2. retune detector thresholds / area priors / merge rules,
3. fine-tune on mouse annotations instead of using human weights as final model.

#### Case B. Adult isolated mouse cardiomyocytes
Example:
- rod-shaped single cells,
- stronger striation,
- higher binucleation rate.

This is still feasible, but the current human-trained detector logic is less likely to transfer cleanly.

Expected changes:
1. DAPI merge heuristics need explicit retuning because binucleation is more common.
2. Size priors and postprocess thresholds must be recomputed.
3. Augmentation should include stronger orientation/aspect-ratio variability matching adult rod-like cells.
4. A human-trained decoder may still help, but mouse-specific fine-tuning is likely necessary.

#### Case C. Mouse tissue sections / histology
Example:
- cross-sections in tissue,
- dense packed myocardium without isolated whole-cell boundaries.

This is **not** a simple transfer of the current project.

Why:
- current pipeline assumes cultured-cell whole-cell instance segmentation,
- nucleus-to-cell assignment becomes weaker in tissue,
- DAPI-centered box prompting may no longer map one nucleus to one visible full cell.

This case would require a larger redesign:
- possibly different prompts,
- different detector logic,
- potentially different target definition.

## 6. What Must Change for Mouse Transfer

### 6.1 Minimum viable adaptation

If the goal is a quick mouse pilot rather than a publishable final model, the minimum changes are:

1. collect a labeled mouse train/val/test split,
2. recompute:
   - nucleus area statistics,
   - cell mask area statistics,
   - edge-filter thresholds,
   - merge-distance priors,
3. retune detection profiles,
4. run `T27a` checkpoint zero-shot only as a reference, not as the final model,
5. fine-tune decoder on mouse data.

### 6.2 Recommended adaptation plan

A more defensible mouse adaptation plan is:

**M0. Zero-shot transfer audit**
- Evaluate current `T27a` directly on mouse data.
- Purpose: quantify raw domain gap.

**M1. Detection-only retune**
- Keep segmentation weights fixed.
- Regenerate DAPI / adaptive thresholds for mouse data.
- Purpose: isolate whether failure is mostly prompt quality.

**M2. Decoder fine-tune on mouse boxes**
- Use GT boxes first.
- Purpose: isolate segmentation-branch transfer ceiling.

**M3. Mouse E2E route**
- Replace GT boxes with mouse detector boxes.
- Purpose: assess deployable performance.

**M4. Optional channel remap study**
- If mouse markers differ, compare:
  - BF-only,
  - `blank + DAPI + BF`,
  - `Actn2 + DAPI + BF`,
  - mouse-specific semantic remap.

### 6.3 What is most likely to break first

The first failure point is likely **detection / box quality**, not the segmentation decoder itself.

Reason:
- the current project already shows a large Oracle vs E2E gap on human hiPSC-CM,
- mouse transfer adds another domain shift on top of that,
- species and preparation differences will perturb nucleus-centered prompt quality before they fully break the segmentation branch.

## 7. Recommended Thesis Positioning

### 7.1 How to write the baseline story strongly

Recommended positioning:

- `Cellpose` shows that off-the-shelf generalist cell segmentation is insufficient.
- `SAM ViT-B` shows that generic promptable segmentation with GT boxes is insufficient.
- `CellSAM pretrained` shows that cell-domain foundation transfer is helpful but still not enough.
- `MedSAM` shows that large-scale medical pretraining is strong, but still not a substitute for cardiomyocyte-specific adaptation.
- `T27a` shows that audited path control plus targeted fine-tuning can outperform all four on the locked human hiPSC-CM task.

### 7.2 What not to overclaim

Do not write:
- "our model is better than all foundation models in general"
- "the model generalizes to mouse cardiomyocytes"

Write instead:
- "our current evidence supports a stronger human hiPSC-cardiomyocyte system on the Allen dataset"
- "mouse transfer is plausible but requires species- and preparation-specific retuning and re-evaluation"

## 8. Actionable Proposal for Follow-up Work

### 8.1 If the next goal is stronger thesis writing

Do this first:
1. keep the main result table centered on `T27a / Cellpose / SAM ViT-B / CellSAM / MedSAM`,
2. use the comparison structure in Section 4 to strengthen the thesis narrative,
3. mention mouse transfer only in Discussion / Future Work unless mouse labels are actually collected.

### 8.2 If the next goal is a mouse pilot study

Do this next:
1. secure a mouse cardiomyocyte dataset with explicit channel metadata,
2. run `M0` zero-shot transfer audit,
3. regenerate detection priors,
4. run `M2` with GT boxes before attempting E2E.

## 9. Questions for A2 Review

1. Should this review be cited in thesis Chapter 2 / Chapter 6 directly, or kept as an internal planning document?
2. If we introduce mouse transfer in the thesis, should it stay in Discussion/Future Work only, or become a formal pilot experiment?
3. If a mouse pilot is started later, should it inherit `T27a` loss unchanged first, or should the first mouse run be a simplified `Dice+BCE(+IoU)` control?

## 10. Sources

### External sources
- SAM: `https://arxiv.org/abs/2304.02643`
- CellSAM: `https://doi.org/10.1038/s41592-025-02879-w`
- CellSAM repo: `https://github.com/vanvalenlab/cellSAM`
- MedSAM repo: `https://github.com/bowang-lab/MedSAM`
- Cellpose docs: `https://cellpose.readthedocs.io/`
- Human / mouse cardiomyocyte difference reviews:
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC8718008/`
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC6390119/`
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC8867505/`
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC8034626/`

### Local project sources
- `docs/paper_writing/paper_preparation.md`
- `docs/paper_writing/chapters/ch5_results.md`
- `docs/technical/update_cellsam.md`
- `docs/technical/cellpose_builtin_models_reference.md`
- `docs/report_2.19.md`

