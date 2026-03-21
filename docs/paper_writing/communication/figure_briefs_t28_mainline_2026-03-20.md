# Figure Briefs (T28 Mainline, Consolidated)

This brief follows the locked writing policy:

- Segmentation mainline: `T28` (dual-seed `val71` mean)
- E2E writing line: `T33g(dapi_cm, candidate_aligned, q35) + T28`
- Positioning: biology-prior selection, not detector-metric-optimal claim

## Recommended layout: Main text 3 figures + Supplementary 2 figures

### Main-text figures

1. **Fig-A3-1 Data and Biology Priors**
2. **Fig-A3-2 End-to-End Overview (high-level only)**
3. **Fig-A3-5 T28 Training and Evaluation Flow**

### Supplementary figures

1. **Fig-A3-3 Released-checkpoint audit schema**
2. **Fig-A3-4 CellFinder principle**

This split removes repeated content between checkpoint structure, prompt source, and training detail while keeping the method story complete.

## Consistency locks (must follow)

1. In **Fig-A3-2**, keep training details as dashed side notes only; the main trunk should be inference chain.
2. In **Fig-A3-5**, title should be:
   - `T28 decoder-only training/evaluation flow (inherits T27a freeze strategy)`
3. Always avoid mixed wording where T27a looks like current mainline.

## Fig-A3-1 Data and Biology Priors (Main)

Purpose:
- Explain domain difficulty and why channel semantics are method-critical.

Must-have elements:
- One representative sample with `BF`, `Actn2`, `DAPI`, `GT mask`.
- Callouts for adhesion, weak boundaries, elongated morphology.
- Prior role tags: `DAPI -> localization prior`, `Actn2 -> cardiomyocyte identity prior`.

Caption draft:
- *Task context on human hiPSC-derived cardiomyocyte microscopy. Brightfield provides deployment-friendly imaging but weak boundaries, while DAPI and Actn2 provide complementary biological priors for localization and cardiomyocyte-specific structural cues.*

## Fig-A3-2 End-to-End Overview (Main)

Purpose:
- One-page system view from input channels to final instance masks.

Must-have elements:
- Input: `[BF, Actn2, DAPI]`.
- Prompt split: `Oracle GT boxes` vs `automatic detector boxes`.
- Detector annotation: `T33g dapi_cm candidate_aligned q35`.
- Segmentation annotation: `CellSAM model_cp` route.
- Output/eval split: `Oracle metrics` vs `E2E metrics`.
- Training details only as dashed side notes (not in main trunk).

Caption draft:
- *Overview of the thesis pipeline. The released CellSAM checkpoint is audited and used through the official `model_cp` path; T28 serves as the sealed segmentation backend, while T33g provides biology-prior-aware detector prompts for end-to-end deployment analysis.*

## Fig-A3-5 T28 Training and Evaluation Flow (Main)

Purpose:
- Operational method figure aligned with Algorithm 1/2.

Title lock:
- `T28 decoder-only training/evaluation flow (inherits T27a freeze strategy)`

Must-have elements:
- Training: load `model_cp` -> freeze image/prompt encoders -> train mask decoder.
- Input: T28 legacy 3-channel order `[BF, Actn2, DAPI]`.
- Inference split: Oracle (`GT box`) vs E2E (`auto box`).
- Reporting lock: `val71 dual-seed mean` as mainline, `test73` as reference tier.

Caption draft:
- *Operational flow for T28 training and deployment. The method keeps the released `model_cp` inference semantics, updates only the mask decoder under the inherited freeze strategy, and evaluates Oracle and end-to-end routes under split-aware reporting.*

## Fig-A3-3 Released-Checkpoint Audit Schema (Supplementary)

Purpose:
- Make implementation-level branch semantics reproducible.

Must-have elements:
- Branches: `model`, `model_cp`, `cellfinder`.
- Official segmentation arrow through `model_cp`.
- `cellfinder` backbone aligns with `model.image_encoder` non-neck body.
- Explicit boundary: verifiable release facts only.

Caption draft:
- *Conceptual audit of the released CellSAM checkpoint. The public release contains separate `model`, `model_cp`, and `cellfinder` branches; segmentation inference follows `model_cp`, while `cellfinder` aligns with the non-neck body of `model.image_encoder`.*

## Fig-A3-4 CellFinder Principle (Supplementary)

Purpose:
- Explain detector branch logic without duplicating pipeline overview.

Must-have elements:
- Detector flow: feature extraction -> candidate query -> box outputs.
- Prior injection point: `dapi_cm` and candidate-aligned supervision.
- Boundary note: detector metrics are auxiliary; final claim uses E2E `F1/PQ`.

Caption draft:
- *Detector-side principle used in the thesis. Candidate generation is enhanced by biology priors (`dapi_cm` and candidate-aligned supervision), while deployment conclusions remain anchored to downstream end-to-end F1/PQ rather than detector metrics alone.*
