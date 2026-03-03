# ⚠️ 已合并至 `paper_preparation.md` §7 (2026-02-25)

> 本文件内容已合并至 `paper_preparation.md` 的 §7 写作规划。
> 后续更新请直接编辑 `paper_preparation.md`，本文件保留仅供历史参考。

---

# 论文写作计划 (ARCHIVED)

## 1. 目标期刊推荐

按难度分 3 级，定位"应用型 + 心肌细胞分割"：

| 等级 | 期刊 | IF | 审稿周期 | 适配理由 | 推荐度 |
|------|------|:---:|---------|---------|:---:|
| **Tier 1** | *Frontiers in Cell and Developmental Biology* | ~5.5 | 6-8 周 | 接受 AI+Cell Biology 交叉应用；OA | ⭐⭐⭐ |
| **Tier 1** | *Bioengineering* (MDPI) | ~4.5 | 4-6 周 | 2025 年已发 SAM 综述；Special Issue 有 Medical AI | ⭐⭐⭐ |
| **Tier 2** | *Scientific Reports* (Nature) | ~4.0 | 8-12 周 | 接受应用型论文，影响力高 | ⭐⭐ |
| **Tier 2** | *IEEE JBHI* | ~7.7 | 8-12 周 | 偏生物+健康信息学，接受 AI 应用 (R1 建议) | ⭐⭐ |
| **Tier 2** | *Computers in Biology and Medicine* | ~7.0 | 8-10 周 | 偏方法+应用结合，IF 更高但竞争也大 | ⭐⭐ |
| **Tier 3** | *Medical Image Analysis* | ~10 | 14-20 周 | 顶刊，需要更强的方法贡献（除非结果非常好） | ⭐ |

> **推荐**：先投 **Frontiers/Bioengineering**（快审+OA+领域匹配），被拒后可转投 Scientific Reports。

---

## 2. 写作工具：OpenAI Prism vs Overleaf

| 特性 | **OpenAI Prism** (新) | **Overleaf** (经典) |
|------|------|------|
| LaTeX 支持 | ✅ 云端 LaTeX | ✅ 云端 LaTeX |
| AI 辅助 | ✅ GPT-5.2 内嵌，**直接在文档内对话** | ❌ 需外部 AI 辅助 |
| 文献检索 | ✅ 自动搜索+插入引用 | ❌ 需手动导入 .bib |
| 协作 | ✅ 实时协作+语音编辑 | ✅ 实时协作 |
| 价格 | ✅ **免费** (ChatGPT 账号) | 免费版有限功能 |
| 模板 | 🟡 新平台，模板可能较少 | ✅ 大量期刊模板 |
| 成熟度 | 🟡 2026.01 刚发布 | ✅ 10年+稳定使用 |

> **建议**：**用 Prism 写初稿**（利用 AI 辅助起草+文献检索），最终版转 Overleaf 用期刊模板排版。或直接在 Prism 里用，因为它本身就是 LaTeX 环境。

---

## 3. 论文大纲 (Application Paper)

### Title (暂定)
*CellSAM Fine-tuning for hiPSC-CM Segmentation with DAPI-guided Detection*

> R1 review: 去掉 "Instance"，精简标题

### Abstract (最后写)

### 1. Introduction (~1.5 pages)
**现在可写**: ✅

| 段落 | 内容 |
|------|------|
| P1 | 心肌细胞分割在药物筛选/心脏病研究中的重要性 |
| P2 | 现有方法的局限：Cellpose/StarDist 为通用细胞设计，不针对心肌细胞特征 |
| P3 | Foundation Model (SAM/CellSAM) 的兴起 + fine-tuning 的必要性 |
| P4 | 我们的贡献 (3 点): (1) 首次微调 CellSAM 用于心肌细胞 (2) DAPI-guided 检测管线 (3) 系统性参数消融 |

### 2. Related Work (~1 page)
**现在可写**: ✅

| 小节 | 关键引用方向 |
|------|------------|
| 2.1 Cell Segmentation | Cellpose, StarDist, HoVerNet |
| 2.2 Foundation Models for Segmentation | SAM, MedSAM, CellSAM, MedicoSAM (2025) |
| 2.3 Cardiomyocyte Analysis | Allen Institute dataset, 现有 hiPSC-CM 分析方法 |

### 3. Method (~3 pages)
**现在可写**: ✅（大部分）

| 小节 | 内容 | 可写性 |
|------|------|:---:|
| 3.1 Dataset | Allen hiPSC-CM; 334/71/73 split; 通道映射 (BF→SAM) | ✅ |
| 3.2 Detection Pipeline | DAPI 核检测 → 框生成 → SAM prompt | ✅ |
| 3.3 SAM Fine-tuning (Phase 1) | Loss 设计: Dice+BCE+Boundary+AJI（Contour 经消融后移除）; 冻结 encoder | ✅ |
| 3.4 Exclusion-Aware Losses (Phase 2) | L_neighbor + L_overlap 设计动机 | ✅ 框架可写，结果待定 |

> **P2 降级预案 (R1 review)**: 若 Fix3 仍不及 P1 (PQ=0.475)，将 §3.4 定位为 "Preliminary Exploration"，不作为主要贡献，移入 Discussion 作为 Future Work。

### 4. Experiments (~2.5 pages)
**大部分需要实验完成后写**

| 小节 | 内容 | 状态 |
|------|------|:---:|
| 4.1 Implementation Details | GPU, epochs, LR, optimizer | ✅ 可写 |
| 4.2 Evaluation Metrics | PQ, Dice, AJI, Detection F1 | ✅ 可写 |
| 4.3 Detection Ablation | E34/E34b 消融结果 | ✅ **已有数据** |
| 4.4 Training Ablation | Phase 1 loss 权重消融 | ⚠️ 需补做 |
| 4.5 Comparison with Baselines | vs CellSAM(原始), MedSAM, Cellpose, StarDist | ❌ **需补做** |
| 4.6 Phase 2 Results | Fix1/Fix2/Fix3 对比 (或 Preliminary Exploration) | ⚠️ 进行中 |

### 5. Results & Discussion (~1.5 pages)
**需要实验完成后写**

### 6. Conclusion (~0.5 pages)
**最后写**

---

## 4. 待补实验清单

| # | 实验 | 优先级 | 预计耗时 | 依赖 |
|---|------|:---:|---------|------|
| **E-B4** | CellSAM 原始 (不微调, test73) | 🔴 | 1h | 已有代码基础 |
| **E-B1** | Cellpose baseline (test73) | 🔴 | 2h | 安装 cellpose |
| **E-B2** | StarDist baseline (test73) | 🔴 | 2h | 安装 stardist |
| **E-B5** | **MedSAM baseline (test73)** | 🔴 | 2h | §2.2 提到须对比 (R1 review) |
| **E-B3** | HoVerNet baseline (test73) | 🟡 | 4h | 模型较重，**放后做** |
| **E-A1** | Phase 1 loss 权重消融 (val71) | 🟡 | 需多次训练 | 已有 config |
| **E-P2** | Phase 2 最终结果 (Fix3?) | 🟡 | 取决于 Fix3 方案 | 需先确定方向 |

> **优先级排序 (R1 review)**: E-B4 → E-B1 → E-B2 → E-B5 (~5h 核心)，HoVerNet 放后。

---

## 5. 写作时间表

### Phase 1: 框架搭建 (本周)
- [ ] 在 Prism / Overleaf 创建项目 + 选模板
- [ ] 写 §1 Introduction
- [ ] 写 §2 Related Work
- [ ] 写 §3 Method (3.1-3.3)
- [ ] 写 §4.1 Implementation Details + §4.2 Metrics

### Phase 2: Baseline 实验 (下周)
- [ ] 跑 E-B4 CellSAM 原始 (~1h)
- [ ] 安装 Cellpose + StarDist，跑 E-B1, E-B2 (~4h)
- [ ] 安装 MedSAM，跑 E-B5 (~2h)
- [ ] 整理结果表格
- [ ] (可选) HoVerNet E-B3 放 Phase 3 或之后

### Phase 3: 填充结果 (实验完成后)
- [ ] 写 §4.3-4.6 实验结果
- [ ] 写 §5 Discussion
- [ ] 画 Figure (可视化对比图)

### Phase 4: 打磨 (投稿前)
- [ ] 写 Abstract
- [ ] 写 §6 Conclusion
- [ ] 全文校对 + 格式检查
