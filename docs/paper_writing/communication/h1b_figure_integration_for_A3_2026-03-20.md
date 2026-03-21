# H1b Figure Integration For A3 (LaTeX-Ready)

## 0) 唯一写入入口（已对齐）

论文正文只写入：

1. `D:/AI/paper/CellSam/docs/paper_writing/overleaf_export/main.tex`
2. `D:/AI/paper/CellSam/docs/paper_writing/overleaf_export/chapters/*.tex`

说明：此前存在 `docs/paper_writing/chapters/*.md` 与其他 md 讨论文档，它们属于草稿/沟通材料，不是最终 Overleaf 主稿入口。

---

## 1) 论文最终图专用目录

已集中到：

`D:/AI/paper/CellSam/docs/paper_writing/overleaf_export/figures/h1b_final_set_20260320/`

---

## 2) 主文使用图（最终）

### Fig-A Oracle Upper Bound

- 文件：`paper_oracle_random3_no_cellpose_20260319.png`
- 建议插入：`chapters/ch5_results.tex`，放在 `\subsection{Oracle-to-End-to-End Gap}` 段落开头（表 5.4 前后均可，建议表后图）。
- 结论：GT box 下 T28 mask refinement 已足够强，E2E瓶颈主要在 detector prompt 质量。
- caption 草稿：
  - *Oracle comparison on random test samples. With GT prompts, T28 yields high-quality masks, indicating that most end-to-end loss comes from detector prompt quality rather than mask refinement capacity.*

### Fig-B E2E Main Qualitative

- 文件：`h1b_e2e_cmp_test_firstk_k3_20260320_031537.png`
- 建议插入：`chapters/ch5_results.tex`，放在 `\subsection{Prompt-Quality-Aware Fine-Tuning}` 前（或该小节第一段后）。
- 结论：`T33g(dapi_cm)+T28` 相对 raw 路径在 cardiomyocyte 场景更稳健；同时展示 GT-v1 标注歧义样本。
- caption 草稿：
  - *End-to-end qualitative comparison on three representative test samples. The T33g(dapi\_cm)+T28 route improves biologically plausible recovery over raw detector prompts while exposing known GT-v1 ambiguity cases.*
- 正文必写免责声明（1句）：
  - *GT-v1 contains likely missing/over-labeled instances in a small subset; qualitative interpretation is therefore reported jointly with metric evidence.*

### Fig-C T33 Training Curves

- 文件：`h1b_training_curves_core_20260320_060926.png`
- 建议插入：`chapters/ch5_results.tex`，放在 detector 演化/`H1bA` 叙述段（可在 `\subsubsection{Why the CellFinder line still underperforms end-to-end}` 后追加小段引图）。
- 结论：
  - `candidate_aligned_f1@0.3`（与 T33f/g 早停一致）显示 candidate-aware 线起点高；
  - `val_f1@0.5` 与 `AP50` 作为辅助监控指标。
- caption 草稿：
  - *Training dynamics of T33 detector variants. Candidate-aligned F1@0.3 is the prior-aligned optimization target, while F1@0.5 and AP50 are auxiliary detector-quality monitors.*

---

## 3) 机制解释图（附录/补充）

用户已决定：**不使用三模型闭环热力图**（pretrained/T33b/T33g 同图）作为论文主图。

当前保留两张更合适的热力图（可放附录）：

### Fig-S1 T28 Encoder Activation Heatmap

- 文件：`t28_attention_heatmap_9797a59f_5500000013_63X_20190807_S2_P16_C5.png`
- 建议插入：`chapters/app_b_exploratory_results.tex`
- 结论：展示 T28 在 BF-replicate 与 3ch 输入下的 encoder 激活差异（机制证据，非主指标）。

### Fig-S2 T28 Box-Outside Probability Heatmap

- 文件：`t28_box_attention_9797a59f_5500000013_63X_20190807_S2_P16_C5_cell10.png`
- 建议插入：`chapters/app_b_exploratory_results.tex`
- 结论：定量展示 no-clipping 条件下 box 外概率质量占比，支撑 box-geometry 与 refinement 关系分析。

---

## 4) 可直接粘贴的 LaTeX 模板

> 路径均相对 `overleaf_export/` 根目录。

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/h1b_final_set_20260320/paper_oracle_random3_no_cellpose_20260319.png}
  \caption{Oracle comparison on random test samples. With GT prompts, T28 yields high-quality masks, indicating that most end-to-end loss comes from detector prompt quality rather than mask refinement capacity.}
  \label{fig:h1b_oracle_random3}
\end{figure*}
```

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/h1b_final_set_20260320/h1b_e2e_cmp_test_firstk_k3_20260320_031537.png}
  \caption{End-to-end qualitative comparison on three representative test samples. The T33g(dapi\_cm)+T28 route improves biologically plausible recovery over raw detector prompts while exposing known GT-v1 ambiguity cases.}
  \label{fig:h1b_e2e_main}
\end{figure*}
```

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/h1b_final_set_20260320/h1b_training_curves_core_20260320_060926.png}
  \caption{Training dynamics of T33 detector variants. Candidate-aligned F1@0.3 is the prior-aligned optimization target, while F1@0.5 and AP50 are auxiliary detector-quality monitors.}
  \label{fig:h1b_t33_curves}
\end{figure}
```

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/h1b_final_set_20260320/t28_attention_heatmap_9797a59f_5500000013_63X_20190807_S2_P16_C5.png}
  \caption{T28 encoder activation heatmap under BF-replicate vs 3-channel input.}
  \label{fig:h1b_t28_encoder_heatmap}
\end{figure}
```

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/h1b_final_set_20260320/t28_box_attention_9797a59f_5500000013_63X_20190807_S2_P16_C5_cell10.png}
  \caption{T28 box-outside probability heatmap (no clipping) for box-geometry analysis.}
  \label{fig:h1b_t28_box_heatmap}
\end{figure}
```

---

## 5) H1b 交接状态（给 A1/A3）

1. 主文图、补充热力图、caption 草稿、章节放置位置均已明确。  
2. 图资源已集中到 `overleaf_export/figures/h1b_final_set_20260320/`。  
3. H1b 论文素材可直接由 A3 在 `overleaf_export` 的 LaTeX 主稿中落版。  
