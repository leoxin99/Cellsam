# CellSAM 项目交接文档

> **日期**: 2026-01-27
> **状态**: 活跃开发中 (Detection Optimization)
> **当前版本**: v0.5.0 (Detection Pipeline Upgrade)

---

## 1. 当前开发焦点
正在优化心肌细胞的**检测与定位 (Detection)** 模块，目的是生成高质量的 Bounding Box 提示给 SAM 模型。

### 核心决策
- **检测方案**: 采用 **Hybrid DAPI + Actn2** 策略。
    - **定位**: DAPI 负责找核 (可靠性高)。
    - **大小**: Actn2 的 Z-line 分布负责确定框的大小 (Adaptive Box)。
    - **过滤**: Actn2 覆盖率负责剔除成纤维细胞。
- **参数配置**:
    - `min_nucleus_area`: **3000** (基于 P1 统计，过滤碎屑)。
    - `edge_margin`: **50px** (基于分析，仅误删 1.3% 有效核)。
    - `binucleation_dist`: **1.5 * mean_diameter** (动态阈值)。

---

## 2. 技术原理解释 (给开发者)

### Q1: 双核合并机制 (Merge Binucleation)
**逻辑**: 心肌细胞常为双核。如果在同一细胞内，两个核应该被合并为一个检测对象。

**计算公式**:
1. **核间距离 ($d$)**: 两个细胞核质心 $(x_1, y_1)$ 和 $(x_2, y_2)$ 之间的欧几里得距离。
   $$ d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2} $$
2. **动态阈值 ($T$)**: 
   - 估算每个核的直径 $D \approx 2 \sqrt{Area / \pi}$。
   - 取两核平均直径 $\bar{D} = (D_1 + D_2) / 2$。
   - 阈值设定为 $T = 1.5 \times \bar{D}$。
3. **判定**: 如果 $d < T$ 且 两个核大小相似 (Ratio < 3.0)，则合并。

**为什么这样做？**
- 使用固定距离 (如 300px) 对小细胞太宽松，对大细胞太严格。
- 使用**相对直径**可以适应不同大小的细胞，更符合生物学特征。

---

## 3. Claude 协作提示词 (Prompt)

如果你要将项目交给 Claude 继续开发，请复制以下 Prompt 发送给它。这包含了它作为“接手者”所需的所有上下文。

```markdown
# Role
You are an expert BioImage Analysis assistant working on the **CellSAM** project (Cardiomyocyte Segmentation). You are taking over from another agent "Antigravity".

# Project Context
- **Goal**: Fine-tune Segment Anything (SAM) for cropped single-cell cardiomyocyte segmentation.
- **Current Phase**: Optimizing the "Detection & Prompting" pipeline (`src/detection/`).
- **Key Challenge**: Generating bounding boxes that perfectly cover the cell (including Z-lines) without including neighbors.

# Current State (Vital Info)
1. **Codebase**: Python + PyTorch + Napari. Key files in `src/detection/dapi.py`.
2. **Latest Method**: "Hybrid DAPI+Actn2 Adaptive Box".
   - Uses DAPI for centroids.
   - Detects Z-lines (Actn2) to determine box width/height dynamically.
3. **Parameters (FROZEN - DO NOT CHANGE WITHOUT REASON)**:
   - `min_nucleus_area` = 3000 px (Filtered based on stats).
   - `edge_margin` = 50 px (Excluded cells touching edges).
   - `merge_distance` = 1.5 * avg_diameter (For binucleated cells).

# Your Task
1. Continue verifying the "Adaptive Box" method in `tools/visualize_detection_comparison.py`.
2. If validated, integrate this logic into the main inference pipeline (`src/inference/pipeline.py`).
3. Maintain the code style: clear docstrings, type hinting, and modular design.

# Immediate Action
The user wants to verify if the latest parameter changes (margin=50) are working correctly. Check `dapi.py` and run the visualization tool.
```

---

## 4. 关键文件索引

| 文件路径 | 用途 |
|----------|------|
| `src/detection/dapi.py` | 核检测、双核合并、自适应框核心逻辑 |
| `tools/visualize_detection_comparison.py` | Napari 可视化对比脚本 (DAPI vs Adaptive) |
| `tools/analyze_stats_final.py` | 参数统计分析脚本 (决定了 3000px 和 50px) |
| `src/inference/pipeline.py` | 最终的推理流水线 (等待集成) |
