# CellSAM 论文 Methods 草稿

> **用途**: 直接用于论文 Methods 部分
> **最后更新**: 2026-01-11

---

## 2. Materials and Methods

### 2.1 Dataset

We used the Allen Cell Collection dataset containing **478 fields** of human induced pluripotent stem cell-derived cardiomyocytes (hiPSC-CMs). Each image is a 10-channel OME-TIFF with the following relevant channels:

| Channel | Content | Usage |
|---------|---------|-------|
| Ch0 | Brightfield | Model input |
| Ch1 | 488nm (α-actinin-2-GFP) | Sarcomere marker |
| Ch4 | DAPI | Nucleus detection |
| Ch9 | Instance segmentation mask | Ground truth |

Images were acquired at 63× magnification with typical dimensions of ~1736×1776 pixels.

### 2.2 Cell Detection

Due to the failure of the original CellFinder detector (F1 = 0.012) on cardiomyocyte morphology, we developed a DAPI-based detection pipeline:

1. **Preprocessing**: Percentile normalization (2nd-98th percentile mapping to 0-1)
2. **Thresholding**: Otsu's method on normalized DAPI channel
3. **Morphological cleanup**: Binary opening (disk r=3), hole filling, small object removal (min_area=500)
4. **Size filtering**: Exclusion of nuclei outside 500-15000 pixel² range
5. **Relative size filtering**: Exclusion of nuclei smaller than 20% of median area
6. **Binucleate merging**: Nuclei within 100px merged as single cell
7. **Edge exclusion**: Cells within 30px of image boundary excluded

Bounding boxes were created by expanding nucleus regions by 6× in each dimension.

### 2.3 Segmentation Model

We fine-tuned the CellSAM model (based on Segment Anything Model architecture) using:

- **Training set**: 50 images with 350 cell instances
- **Loss function**: Combined loss = 0.5×Dice + 0.5×BCE with dynamic pos_weight
- **Optimizer**: Adam, learning rate 1×10⁻⁴
- **Epochs**: 50
- **Data augmentation**: Random flips, rotations, brightness/contrast adjustments

### 2.4 Instance Segmentation Pipeline

For each detected cell:
1. Brightfield image resized to 1024×1024 and converted to RGB
2. Bounding box scaled proportionally
3. SAM inference with box prompt
4. Post-processing: morphological closing (disk r=5), hole filling, largest connected component extraction
5. Instance ID assignment to non-overlapping pixels

### 2.5 Evaluation Metrics

**Detection metrics** (box-level):
- Precision, Recall, F1-score at IoU threshold 0.5

**Segmentation metrics** (pixel-level):
- Overall Dice coefficient
- Per-cell Dice coefficient

**Planned additional metrics** (see Section 9 of progress report):
- Panoptic Quality (PQ = SQ × RQ)
- Aggregated Jaccard Index (AJI)
- Boundary IoU
- Rand Index

---

## 3. Results 数据来源索引

| Figure/Table | 数据来源 | 实验ID |
|--------------|---------|--------|
| Detection comparison | E02, E03 | CellFinder vs DAPI |
| Segmentation results | E04, E05 | Pixel vs Instance |
| Ablation: Watershed | E06 | 分水岭实验 |
| Overall pipeline | exp_20260109_204227 | 10样本测试 |

---

## 附录：论文用数值汇总

### Detection Performance
```
CellFinder: P=0.009, R=0.016, F1=0.012
DAPI-based: P=0.708, R=0.797, F1=0.750
```

### Segmentation Performance (10 test samples)
```
Pixel-level:   Overall Dice=0.5757, Mean Cell Dice=0.7623
Instance-level: Overall Dice=0.7066
```

### Model Training
```
Training samples: 50 (with 350 cell instances)
Validation Dice: 0.52
Test Dice: 0.71-0.78 (depending on metric)
```
