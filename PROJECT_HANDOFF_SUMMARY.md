# CellSAM 项目交接汇总 (Project Handoff Summary)

**最后更新时间**: 2026-01-07
**项目目标**: 实现 Allen hiPSC-CM 心肌细胞的全自动、高精度分割。

---

## 1. 项目核心架构

### 1.1 模型架构 (CellSAM)
我们采用 **解耦 (Decoupled)** 的训练与推理策略：

*   **Image Encoder (ViT-H)**: ❄️ **冻结**。利用 SA-1B 预训练权重提取通用视觉特征。
*   **Prompt Encoder**: ❄️ **冻结**。负责将边界框编码为位置提示。
*   **Mask Decoder**: 🔥 **训练**。这是唯一微调的组件，学习心肌细胞的特殊纹理和边界特征。
*   **CellFinder**: ❄️ **冻结**。预训练的细胞检测头，用于推理阶段自动生成提示框。

### 1.2 训练策略 (Training Strategy)
*   **输入**: 图像 + **GT 边界框** (从标注掩膜动态生成)。
*   **逻辑**: "提示工程"训练。告诉模型"这里肯定有细胞"，强迫它学习分割边界。
*   **为何不用 CellFinder 训练?**: 避免检测器的噪声干扰分割器的训练。
*   **为何不用 DAPI 训练?**: 心肌细胞尺寸大且常双核，DAPI 只能定位核，无法代表细胞体。

### 1.3 推理策略 (Inference Strategy)
*   **全自动流程**: 输入未见过的明场图像 → **CellFinder** 自动检测所有细胞框 → **Mask Decoder** 精细分割每个框 → 合并输出。
*   **兜底方案**: 若 CellFinder 效果不佳，可使用传统图像处理生成框，或人工少许点提示。

---

## 2. 当前项目状态

### 2.1 目录结构 (`d:\AI\paper\CellSam`)
```text
CellSam/
├── .agent/                     # 工作流配置
├── .claude/                    # Claude Code 配置
├── .git/
├── .gitignore
├── PROJECT_HANDOFF_SUMMARY.md  # 核心交接文档
├── token.txt                   # API Token
│
├── src/                        # 核心代码模块
│   └── augmented_dataset.py    # 增强数据集类
│
├── data/                       # 数据文件夹
│   ├── raw/allen_segmented_fields_full/  # 478 张原始 TIFF
│   ├── processed/              # (待生成) 训练用 NPY
│   ├── scripts/
│   │   ├── download_full_segmented.py
│   │   └── extract_expanded_pairs.py
│   └── manifest.csv
│
├── tools/                      # 工具脚本
│   ├── view_annotation_tiff.py
│   ├── view_test_results.py
│   └── view_with_cellsam.py
│
├── docs/                       # 文档
│   ├── CellSAM_run_tutorial.md
│   └── foundation-SAM.pdf
│
├── checkpoints/                # 模型权重
├── cellSAM_source/             # CellSAM 框架
│
├── train_expanded.py           # 主训练入口
├── test_model.py               # 测试入口
└── run_cellsam.py              # 推理入口
```

### 2.2 数据集
*   **来源**: Allen Institute `2d_segmented_fields`
*   **总量**: 478 张 (10通道 TIFF)
*   **通道**: Ch0=明场 (Input), Ch9=实例分割掩膜 (GT)
*   **预处理**: 百分位归一化 + Padding (保持长宽比) + Resize to 1024x1024

---

## 3. 详细实施细节

### 3.1 训练配置
*   **Loss Function**: Per-cell Dice + BCE (每个细胞单独计算损失)
*   **Optimizer**: AdamW, LR=1e-4, Weight Decay=0.01
*   **Augmentation**: Albumentations (RandomRotate90, Flip, ElasticTransform, BrightnessContrast)
*   **Epochs**: 50 (计划)
*   **关键改进**: 使用 `cell_ids` 将每个预测掩膜与其对应的 GT 细胞区域匹配，而非合并后比较

### 3.2 评估指标体系
1.  **像素级 (Pixel-level)**: Dice Score, IoU (衡量整体前景重叠)
2.  **实例级 (Instance-level)**: 
    *   **Precision/Recall/F1**: 衡量检测准确度 (IoU > 0.5 视为 TP)
    *   **Instance Dice**: 衡量单个细胞的分割质量

---

## 4. 后续步骤 (Action Plan)

接下来的工作流建议：

1.  **数据提取**: 运行脚本处理 `data/raw` 下的 478 张图像，生成训练对到 `data/processed`。
2.  **代码更新**: 修改 `train_expanded.py` 以适配新的数据路径和实例级评估代码。
3.  **全面训练**: 在完整数据集上运行 50 Epochs。
4.  **端到端测试**: 使用 `run_cellsam.py` 测试新图像的自动化分割效果。

---

## 5. 常见问题 (FAQ)

*   **Q: 如何处理 CellFinder 漏检?**
    *   A: 可以在预处理阶段加入传统形态学检测作为辅助提示，或者微调检测头(Tier 2方案)。
*   **Q: 为什么不用 LoRA?**
    *   A: 当前阶段 Mask Decoder 微调已足够有效且高效。LoRA 可作为后期提升手段。
*   **Q: GT 框是怎么来的?**
    *   A: `skimage.measure.regionprops(mask).bbox`

此文件旨在帮助 Claude Code 或其他协作者快速接手项目。
