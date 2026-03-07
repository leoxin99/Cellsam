# Cellpose Built-in Models Reference

> **作者**: Codex  
> **状态**: 🟢 Active  
> **最后更新**: 2026-03-07  
> **用途**: 统一说明 Cellpose 官方 built-in models 的数据来源、任务定位、与心肌细胞项目的相关性

---

## 1. 结论先行

1. 我们当前项目里用 `cyto3` 作为 Cellpose baseline 是合理的。
   - 它是 Cellpose 官方当前最常用的 **generalist whole-cell** built-in model。
   - CellSAM 论文公开 benchmark 也明确把 pretrained generalist Cellpose baseline 写成 `cyto3`。

2. Cellpose 确实公开了一批 dataset-specific built-in models，但**没有一个是专门针对心肌细胞**。

3. 对我们的心肌细胞任务:
   - **主 baseline**: `cyto3`
   - **可做补充探索**: `livecell_cp3`, `tissuenet_cp3`
   - **不建议作为主 baseline**: `nuclei`, `yeast_*`, `bact_*`, `deepbacs_cp3`

---

## 2. 证据来源

### 2.1 CellSAM 公开 evaluation wrapper

`cellSAM_source/paper_evaluation/models.py` 当前内置了以下 Cellpose model types:

- `nuclei`
- `cyto`
- `cyto2`
- `cyto3`
- `tissuenet`
- `livecell`
- `tissuenet_cp3`
- `livecell_cp3`
- `yeast_PhC_cp3`
- `yeast_BF_cp3`
- `bact_phase_cp3`
- `bact_fluor_cp3`
- `deepbacs_cp3`
- `cyto2_cp3`

代码证据:
- `cellSAM_source/paper_evaluation/models.py:43`

### 2.2 Cellpose 官方文档

Cellpose 官方文档说明:
- `cyto3` 是主 built-in generalist 模型。
- `tissuenet_cp3`, `livecell_cp3`, `yeast_*_cp3`, `bact_*_cp3`, `deepbacs_cp3`, `cyto2_cp3` 是 **dataset-specific models trained on one of the 9 datasets in the Cellpose3 paper**。

官方文档:
- https://cellpose.readthedocs.io/en/v3.1.1.1/models.html

### 2.3 CellSAM 论文对 Cellpose baseline 的写法

CellSAM 论文主文/补充材料明确区分:
- pretrained generalist Cellpose model (`cyto3`)
- internally trained generalist Cellpose
- internally trained specialist Cellpose

本地证据:
- `docs/temp_reviews/methods_page_3.txt:61`
- `docs/temp_reviews/methods_page_11.txt:42`

---

## 3. Built-in Models 技术表

| Model | 类型 | 官方来源 / 数据集 | 主要任务 / 域 | CellSAM wrapper 默认 channels | 对心肌细胞是否适合 | 说明 |
|------|------|------|------|------|------|------|
| `nuclei` | Generalist | Cellpose 通用核模型 | nucleus segmentation | `[3,0]` | 不适合作为主 baseline | 只分核，不分 whole-cell |
| `cyto` | Generalist | Cellpose 早期通用模型 | whole-cell / cytoplasm | `[3,2]` | 可跑，但不推荐主用 | 旧版通用模型 |
| `cyto2` | Generalist | Cellpose 2 通用模型 | whole-cell / cytoplasm | `[3,2]` | 可跑，但优先级低于 `cyto3` | 比 `cyto` 更新 |
| `cyto3` | Generalist | Cellpose 3 主通用模型 | whole-cell / cytoplasm | `[3,2]` | **最适合作为主 baseline** | CellSAM paper-aligned baseline 就是它 |
| `tissuenet` | Legacy dataset-specific / general tissue | TissueNet | fluorescence whole-cell | `[3,2]` | 可做补充，不适合主 baseline | 旧一代 TissueNet model |
| `livecell` | Legacy dataset-specific / general live-cell | LIVECell | phase-contrast whole-cell | `[3,0]` | 可做补充，不适合主 baseline | 旧一代 LIVECell model |
| `tissuenet_cp3` | Dataset-specific | TissueNet | fluorescence whole-cell | `[3,2]` | 可做补充探索 | 与心肌细胞显微/形态域不完全一致 |
| `livecell_cp3` | Dataset-specific | LIVECell | label-free / phase whole-cell | `[3,0]` | **补充探索里最值得试** | 比菌/酵母类更接近哺乳动物细胞 |
| `yeast_PhC_cp3` | Dataset-specific | YEAZ | yeast, phase contrast | `[3,2]` | 不适合 | 目标域错误 |
| `yeast_BF_cp3` | Dataset-specific | YEAZ | yeast, brightfield | `[3,2]` | 不适合 | 目标域错误 |
| `bact_phase_cp3` | Dataset-specific | Omnipose | bacteria, phase | `[3,2]` | 不适合 | 目标域错误 |
| `bact_fluor_cp3` | Dataset-specific | Omnipose | bacteria, fluorescence | `[3,2]` | 不适合 | 目标域错误 |
| `deepbacs_cp3` | Dataset-specific | DeepBacs | bacteria | `[3,2]` | 不适合 | 目标域错误 |
| `cyto2_cp3` | Dataset-specific | Cellpose dataset | whole-cell | `[3,2]` | 可做补充，不作为主 baseline | 名字接近 generalist，但官方把它归到 dataset-specific |

---

## 4. 为什么项目主 baseline 用 `cyto3`

原因有三条:

1. **与 CellSAM 论文口径一致**
   - 论文明确把 pretrained generalist Cellpose baseline 写成 `cyto3`。
   - 证据: `docs/temp_reviews/methods_page_3.txt:61`

2. **与 CellSAM public evaluation 代码一致**
   - `eval_main.py` 默认就是 `model_type='cyto3'`。
   - 证据: `cellSAM_source/paper_evaluation/eval_main.py:79`

3. **对我们的任务最公平**
   - 心肌细胞任务需要的是 whole-cell baseline，不是 nucleus-only baseline。
   - `cyto3` 是官方当前最标准、最通用、最容易解释的 whole-cell baseline。

因此:
- 论文主表 / 正式 baseline: 用 `cyto3`
- 若要做扩展实验，可额外测试 `livecell_cp3` 和 `tissuenet_cp3`

---

## 5. 哪些模型最值得在心肌细胞任务上做补充实验

按优先级建议:

1. `cyto3`
   - 主 baseline
   - 原因: 官方 generalist, paper-aligned, 最稳妥

2. `livecell_cp3`
   - 最值得做 supplementary
   - 原因: 仍然是哺乳动物 whole-cell segmentation 场景，比酵母/细菌更接近我们的目标域

3. `tissuenet_cp3`
   - 可做 supplementary
   - 原因: 也是 whole-cell segmentation，但偏 tissue / fluorescence

不建议优先做:
- `nuclei`: 只分核
- `yeast_*`: 形态和域都错
- `bact_*`: 形态和尺度都错
- `deepbacs_cp3`: 细菌数据

---

## 6. 当前项目应如何写文档口径

建议统一写法:

- **主 baseline**: `Cellpose cyto3 (paper-aligned)`
- **补充探索**: `Cellpose livecell_cp3`, `Cellpose tissuenet_cp3`
- **避免写法**:
  - “Cellpose specialist baseline”  
    除非你明确说明是**哪一个公开 built-in**，或者是你们自己重训的 specialist

因为论文里的:
- internally trained specialist Cellpose

并不等于当前公开仓库里任意一个 built-in name。

---

## 7. 和 CellSAM built-in 的对应关系

Cellpose 有一组相对清晰的 built-in model zoo。

CellSAM 没有对应的 per-dataset built-in zoo。当前公开 `get_model()` 只有:
- `cellsam_general`
- `cellsam_extra`

代码证据:
- `cellSAM_source/cellSAM/model.py:50`
- `cellSAM_source/cellSAM/model.py:62`

本地缓存上，这两个模型权重来自同一个 archive:
- `~/.deepcell/models/cellsam_v1.2/cellsam_general.pt`
- `~/.deepcell/models/cellsam_v1.2/cellsam_extra.pt`

公开代码只写清楚:
- `cellsam_general`: trained only on datasets referenced in the publication
- `cellsam_extra`: incorporates additional datasets, recommended beyond paper-covered domains

但**没有公开列出**:
- `cellsam_extra` 具体多了哪些数据集
- 与 `cellsam_general` 的逐数据集、逐任务差别
- per-dataset specialist CellSAM 的公开 built-in 列表

因此当前更严格的项目口径应是:

| 模型族 | 公开 built-in 体系 |
|------|------|
| Cellpose | 有较清晰的 generalist + dataset-specific built-in zoo |
| CellSAM | 公开接口只有 `general` / `extra` 两档，没有公开 per-dataset specialist zoo |

---

## 8. 对当前项目的直接建议

1. Cellpose 主 baseline 继续用 `cyto3`
2. 如果要补充“公开专项模型探索”，优先顺序:
   - `livecell_cp3`
   - `tissuenet_cp3`
3. 不要把 `nuclei` 当成 whole-cell baseline
4. 不要把 CellSAM 的 `cellsam_extra` 写成“specialist model”
5. 若论文里写 specialist，必须区分:
   - CellSAM 论文里的 internally trained specialist
   - Cellpose 论文里的 internally trained specialist
   - 当前公开 built-in models

