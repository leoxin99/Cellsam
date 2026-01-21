# 实验记录 Experiment Log

**实验时间**: 20260115_014117

**模型**: `d:/AI/paper/CellSam/checkpoints/boundary_20260111_012636/best_model.pt`

**测试样本数**: 10

---

## 总体结果 Overall Results

| 指标 | 值 |
|------|----|
| Mean Dice | **0.7467** |
| Total GT Cells | 106 |
| Total Pred Cells | 126 |

---

## 逐样本结果 Per-Sample Results

| # | Sample ID | GT | Pred | Dice |
|---|-----------|----:|-----:|------|
| 1 | cf4fb0e8_5500000013_63X_20190807_S1 | 10 | 12 | 0.6993 |
| 2 | 3a3cf60a_5500000014_63X_20190816_S2 | 10 | 8 | 0.9067 |
| 3 | 27e55ff3_5500000013_63X_20190807_S1 | 13 | 23 | 0.5984 |
| 4 | ec4c125c_5500000013_63X_20190807_S1 | 6 | 10 | 0.5085 |
| 5 | 60f3d143_5500000014_63X_20190816_S2 | 8 | 8 | 0.7872 |
| 6 | 5c2b8632_5500000013_63X_20190807_S2 | 11 | 12 | 0.7591 |
| 7 | 570acc96_5500000013_63X_20190807_S1 | 10 | 12 | 0.8124 |
| 8 | 43283e18_5500000013_63X_20190807_S1 | 18 | 16 | 0.8560 |
| 9 | ebfc8c4d_5500000013_63X_20190807_S1 | 8 | 15 | 0.6904 |
| 10 | 39531263_5500000013_63X_20190807_S1 | 12 | 10 | 0.8495 |

---

## 实验配置 Configuration

- Detection: DAPI-based nucleus detection
- Segmentation: CellSAM (fine-tuned)
- Post-processing: binary_closing, fill_holes, largest_component
