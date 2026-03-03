# Chapter 5: Results and Analysis

## 5.1 Main Results

Table 5.1 presents the Oracle evaluation results on the test set (73 images) for our approach and all baseline methods.

**Table 5.1: Comparison with existing methods (Oracle evaluation, test73)**

| Method | PQ@0.5 | BM-Dice | AJI | Note |
|--------|:------:|:-------:|:---:|------|
| CellSAM (pretrained) | 0.434 | 0.682 | 0.499 | Official inference path, no fine-tuning |
| SAM ViT-B | 0.286 | 0.631 | 0.440 | No cell-specific training |
| Cellpose | — | — | — | Default pretrained |
| StarDist | — | — | — | Default pretrained |
| SAMCell | — | — | — | Pretrained weights |
| Phase 1 (ours, decoder-only) | 0.464 | 0.695 | 0.519 | posw=2, 50 epochs |
| **Best Config (ours)** | **0.484** | **0.720** | **0.570** | **posw=10, 80 epochs, 4 runs mean** |
| T18-C (ours, 3-channel) | 0.500 | 0.725 | 0.574 | BF+DAPI+Actn2, seed=42 |
| MedSAM (reference) | 0.576 | 0.771 | 0.634 | Upper bound |

The original CellSAM achieves PQ=0.434 on cardiomyocytes using its official inference path (including Stage 2 aligned weights, PercentileThreshold preprocessing, and IoU filtering). Our Best Config fine-tuning achieves PQ=0.484, a +5.0pp improvement over the pretrained CellSAM baseline. This substantially outperforms SAM ViT-B (PQ=0.286) while narrowing the gap to MedSAM (PQ=0.576), which benefits from training on 1.5M medical images. ⚠️ Note: T25 re-training (from official weights) pending — results may change.


## 5.2 Loss Function Ablation

We conduct a systematic ablation study on the loss function components, training each configuration with two random seeds (42 and 123) and reporting the mean. All ablations start from the Phase 1 configuration (posw=2) and modify one component at a time. Results are evaluated in Oracle mode on test73.

**Table 5.2: Loss component ablation (Oracle, test73, mean of 2 seeds)**

| Configuration | PQ | BM-Dice | AJI | ΔPQ |
|--------------|:---:|:-------:|:---:|:---:|
| Full (Phase 1, posw=2) | 0.453 | 0.707 | 0.550 | — |
| BCE+Dice only | 0.459 | 0.711 | 0.554 | +0.6 |
| w/o BoundaryLoss | 0.454 | 0.708 | 0.554 | +0.1 |
| **w/o ContourLoss** | **0.476** | **0.718** | **0.564** | **+2.3** |
| w/o AJI Loss | 0.459 | 0.710 | 0.554 | +0.6 |
| w/o PQ early stopping | 0.459 | 0.710 | 0.555 | +0.6 |
| **pos\_weight=10** | **0.494** | **0.724** | **0.573** | **+4.1** |

Two findings stand out with high confidence:

1. **ContourLoss is harmful** (ΔPQ = +2.3pp when removed). ContourLoss was designed to penalize predictions far from ground-truth contours; however, for cardiomyocytes with irregular and elongated shapes, the distance-field-based penalty interferes with the model's ability to learn flexible boundary representations. Removing it consistently improves all metrics across both seeds.

2. **pos\_weight=10 is critical** (ΔPQ = +4.1pp). Cardiomyocyte images exhibit severe foreground-background imbalance: the bounding box region is dominated by background pixels, as cells typically occupy only 30-50% of their bounding box area due to irregular shapes. Increasing the BCE positive weight from 2 to 10 addresses this imbalance, providing the single largest improvement in our ablation.

These two findings led to our **Best Config**: posw=10, ContourLoss removed, all other components retained. The combined effect is PQ=0.484, an improvement of +3.1pp over Phase 1 (PQ=0.453 in ablation) and +2.0pp over Phase 1's best reported result (PQ=0.464).


## 5.3 Multi-Channel Input Ablation

To investigate whether additional fluorescence channels improve segmentation, we compare three input configurations, all using the Best Config loss settings. We also include a BF-only control trained for the same additional epochs to isolate the effect of multi-channel information from extended training.

**Table 5.3: Input channel ablation (Oracle, test73)**

| Configuration | Input Channels | Adapter | PQ | BM-Dice | AJI |
|--------------|---------------|:-------:|:---:|:-------:|:---:|
| Best Config (BF) | BF×3 | — | 0.484 | 0.720 | 0.570 |
| BF continued | BF×3 | — | 0.491 | 0.722 | 0.571 |
| T18-A (BF+Actn2) | R=BF, G=Actn2 | 2→3ch | 0.496 | 0.724 | 0.573 |
| **T18-C (3-channel)** | R=BF, G=Actn2, B=DAPI | 3→3ch | **0.500** | **0.725** | **0.574** |

The three-channel configuration (T18-C) achieves PQ=0.500, a +1.6pp improvement over Best Config. However, the BF continued training control reaches PQ=0.491 (+0.7pp), indicating that part of the improvement comes from additional training epochs. The **net multi-channel effect** is approximately +0.9pp PQ, suggesting that fluorescence channels (α-Actinin2 for cytoplasm structure, DAPI for nuclear position) provide modest but measurable additional information for boundary delineation.

> Note: T18 results are from seed=42 only; seed=123 results pending.


## 5.4 Box Clipping Ablation

SAM generates predictions across the entire image, not just within the bounding box prompt. We evaluate the effect of our box clipping strategy, which zeros out predictions outside an expanded bounding box region (10% expansion of box width and height).

**Table 5.4: Box clipping ablation (Oracle, test73, mean of 2 seeds)**

| Configuration | PQ | BM-Dice | AJI | ΔPQ |
|--------------|:---:|:-------:|:---:|:---:|
| **With box clipping** | **0.453** | **0.707** | **0.550** | — |
| Without box clipping | 0.437 | 0.703 | 0.545 | −1.6 |

Removing box clipping reduces PQ by 1.6pp. Without clipping, the model's predictions extend beyond the target cell's region, leading to increased overlap with neighboring cells and higher false positive rates. This confirms that constraining predictions to the box neighborhood is beneficial for dense cell segmentation scenarios.


## 5.5 Training Curves Analysis

Figure 5.1 compares the training dynamics of Phase 1 and Best Config (both on L4 GPU).

<!-- TODO: embed figures/training_curves_comparison.png -->
*Figure 5.1: Validation metrics during training. Phase 1 (blue) vs Best Config (red). ★ marks the best epoch; dashed lines indicate early stopping. Top-left: training loss; top-right: validation PQ@0.5; bottom-left: BM-1to1 Dice; bottom-right: semantic Dice.*

Key observations:

1. **Faster convergence**: Best Config (red) achieves lower training loss (~0.09) than Phase 1 (~0.28), indicating that the pos\_weight=10 correction enables more effective gradient signal for foreground pixels.

2. **Higher peak performance**: Best Config reaches its best validation PQ=0.508 at epoch 39, compared to Phase 1's best PQ=0.475 at epoch 49. The improvement is consistent across all four metrics.

3. **GPU consistency**: A supplementary comparison of Best Config on A100 vs L4 (Figure S1) shows highly overlapping curves, confirming that performance differences between GPU types are negligible for this model.

> Note: Validation PQ (0.508) differs from test PQ (0.484) due to the normal validation-to-test generalization gap.


## 5.6 End-to-End Evaluation

To assess real-world deployment feasibility, we evaluate our pipeline with DAPI-detected bounding boxes replacing ground-truth boxes.

**Table 5.5: Oracle vs End-to-End evaluation (Phase 1, test73)**

| Evaluation Mode | BM-Dice | PQ@0.5 | AJI |
|----------------|:-------:|:------:|:---:|
| Oracle (GT boxes) | 0.695 | 0.464 | 0.519 |
| E2E (DAPI detection) | 0.545 | 0.172 | 0.318 |
| **Gap** | **−0.150** | **−0.292** | **−0.201** |

The Oracle-to-E2E gap is substantial (ΔPQ = −29.2pp), indicating that detection quality is the primary bottleneck. Our DAPI detector achieves F1=0.803 (IoU≥0.3 on test73), which means approximately 20% of cells are missed or incorrectly localized, directly degrading downstream segmentation. This motivates future work on improving the detection pipeline, potentially through learned detection models such as CellFinder.


## 5.7 LoRA Encoder Fine-tuning

Following the SAMed methodology (Zhang et al.), we apply Low-Rank Adaptation (LoRA) to the Query and Value projection matrices of all 12 ViT-B encoder blocks. Two rank configurations are evaluated, each with two random seeds. LoRA training starts from the Best Config checkpoint and uses gradient checkpointing to fit within the L4 GPU's 24 GB VRAM budget.

**Table 5.7: LoRA encoder fine-tuning results (Oracle, test73, mean of 2 seeds)**

| Configuration | LoRA Params | PQ | BM-Dice | AJI | Best Epoch | ΔPQ vs Best Config |
|--------------|:-----------:|:---:|:-------:|:---:|:----------:|:------------------:|
| Best Config (baseline) | 0 | 0.484 | 0.720 | 0.570 | — | — |
| **LoRA rank=4** | 147K (0.17%) | 0.483 | 0.720 | 0.569 | 13 / 8 | −0.1 |
| **LoRA rank=8** | 295K (0.33%) | **0.494** | **0.725** | **0.578** | 22 / 2 | **+1.0** |
| MedSAM (reference) | — | 0.576 | 0.771 | 0.634 | — | — |

**Per-seed breakdown:**

| Configuration | Seed | PQ | BM-Dice | AJI | Sem-Dice | Best Epoch |
|--------------|:----:|:---:|:-------:|:---:|:--------:|:----------:|
| LoRA r4 | 42 | 0.483 | 0.718 | 0.567 | 0.793 | 13 |
| LoRA r4 | 123 | 0.483 | 0.722 | 0.571 | 0.802 | 8 |
| LoRA r8 | 42 | **0.501** | 0.726 | 0.576 | 0.803 | 22 |
| LoRA r8 | 123 | 0.488 | 0.724 | 0.581 | 0.817 | 2 |

**Analysis:**

LoRA encoder fine-tuning provides preliminary evidence that domain-specific encoder adaptation has potential for cardiomyocyte segmentation, but the improvement is constrained by the limited dataset size.

1. **LoRA rank=4 matches but does not exceed Best Config** (PQ=0.483 vs 0.484). With only 147K additional parameters (0.17% of the encoder), rank=4 provides insufficient capacity for meaningful domain adaptation, essentially learning a near-identity transformation. This confirms that the encoder requires more expressive modifications to capture cardiomyocyte-specific features.

2. **LoRA rank=8 improves all metrics** (ΔPQ=+1.0pp, ΔBM-Dice=+0.5pp, ΔAJI=+0.8pp). Adding just 295K parameters (0.33% of the encoder), rank=8 achieves PQ=0.494, narrowing the gap to MedSAM from 9.2pp to 8.2pp — an 11% reduction. The best single run (seed=42, PQ=0.501) breaks the PQ=0.5 barrier for the first time with our approach.

3. **Promising but unstable convergence**: The best epochs vary dramatically across seeds (epoch 2 to epoch 22 out of 80), indicating that the limited dataset size (334 training images) constrains stable encoder adaptation. The rank=8 seed=123 run achieves peak performance at epoch 2, meaning the model's best state is essentially the starting point with minimal LoRA modification — further training leads to overfitting. This high seed variance (PQ=0.501 vs 0.488) suggests that the optimization landscape for encoder adaptation is sensitive to initialization under data-scarce conditions.

4. **Implications**: The r4→r8 trend (Δ0→+1.0pp) suggests that higher ranks might yield further improvements, but the overfitting observed at r8 indicates that simply increasing rank without additional regularization or training data would likely exacerbate instability. Future work should explore larger LoRA ranks combined with stronger regularization (e.g., dropout, lower learning rates), as well as neck fine-tuning to mitigate the feature distribution shift between the adapted encoder and the frozen neck layers.


## 5.8 Negative Result: Neighbor and Overlap Exclusion Losses

In a preliminary exploration (Phase 2-A), we designed two additional loss components to address inter-cell boundary confusion:

- **L\_neighbor**: Penalizes predictions that overlap with neighboring cells' ground-truth masks, encouraging each prediction to respect cell boundaries.
- **L\_overlap**: Penalizes high-confidence predictions in regions where multiple cells' predictions overlap, reducing conflict zones.

**Table 5.6: Neighbor/Overlap loss experiments (Oracle, test73)**

| Configuration | N/O Weights | Delay | PQ | ΔPQ vs Phase 1 |
|--------------|------------|:-----:|:---:|:-----------:|
| Phase 1 (baseline) | — | — | 0.475 | — |
| Fix1 | N=0.3, O=0.1 | 0 | 0.232 | −51% |
| Fix2 | N=0.1, O=0.05 | 0 | 0.393 | −17% |
| Fix3 | N=0.1, O=0.05 | 10 epochs | 0.466 | −2% |

All three configurations degrade performance. Even with delayed activation (Fix3, where N/O losses activate only after epoch 10), the best PQ occurs at epoch 3 (before N/O activation) and monotonically decreases thereafter. This suggests that the exclusion losses create conflicting gradient signals with the primary segmentation losses: while the base loss encourages the model to predict the target cell, the exclusion losses penalize any overlap with neighbors, leading to overly conservative predictions that shrink away from cell boundaries.

This negative result is informative: it demonstrates that naive repulsion-based losses do not effectively address inter-cell boundary confusion in dense cell segmentation. Alternative approaches, such as contrastive learning between neighboring instances or boundary-aware attention mechanisms, may be more promising directions for future work.
