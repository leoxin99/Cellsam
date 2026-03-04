# [Codex | 2026-03-04] T31 Cellpose Baseline 重跑方案

## 1. 背景

当前仓库里的历史 Cellpose baseline 不适合作为正式论文对比基线，原因已可由代码和结果直接确认：

1. `tools/baseline_eval.py` 已标记为 deprecated，且 Cellpose 路径只喂 BF 单通道灰度图。
   - `tools/baseline_eval.py:1`
   - `tools/baseline_eval.py:147`
   - `tools/baseline_eval.py:148`
   - `tools/baseline_eval.py:152`
2. CellSAM 官方公开评估代码使用的是 Cellpose generalist `cyto3`，并显式指定 `channels=[3,2]`。
   - `cellSAM_source/paper_evaluation/eval_main.py:85`
   - `cellSAM_source/paper_evaluation/models.py:47`
   - `cellSAM_source/paper_evaluation/models.py:92`
3. 当前历史结果表现为严重过分割，说明实现与任务口径同时失配。
   - `experiments/baseline_comparison/results_combined.json`
   - `cellpose_v4`: `PQ=0.000211`, `FP=255.44/img`, `n_pred=255.45/img`
   - `cellpose_v4_d200`: `PQ=0.001944`, `FP=16.51/img`

因此需要补做一轮按 CellSAM 论文口径对齐的 Cellpose baseline。

## 2. 目标

T31 只回答两个问题：

1. 如果按 CellSAM 论文/官方评估代码的方式使用 Cellpose，它在当前 Allen 心肌细胞数据上的真实水平是多少。
2. 之前 Cellpose 几乎完全失败，到底是模型本身不适合，还是历史 baseline 实现错配导致的。

## 3. 对齐口径

### 3.1 CellSAM 官方口径

CellSAM 官方公开评估代码中，Cellpose 默认是：

- `model_type='cyto3'`
- `channels=[3,2]`
- 图像先做逐通道归一化
- 最终输出论文口径指标 `F1` 和 `Recall`

代码证据：
- `cellSAM_source/paper_evaluation/eval_main.py:29`
- `cellSAM_source/paper_evaluation/eval_main.py:85`
- `cellSAM_source/paper_evaluation/models.py:38`
- `cellSAM_source/paper_evaluation/models.py:47`
- `cellSAM_source/paper_evaluation/models.py:91`
- `cellSAM_source/paper_evaluation/cpm.py:43`
- `cellSAM_source/paper_evaluation/cpm.py:60`
- `cellSAM_source/paper_evaluation/cpm.py:62`

### 3.2 当前项目上的映射

当前 Allen 数据的有效原始通道是：

- `BF = Ch0`
- `Actn2 = Ch1`
- `DAPI = Ch4`

项目处理后的 3 通道数组口径是：

- `image[0] = BF`
- `image[1] = DAPI`
- `image[2] = Actn2`

证据：
- `docs/dataset_parameters.md:48`
- `docs/dataset_parameters.md:52`
- `docs/dataset_parameters.md:53`
- `docs/dataset_parameters.md:54`
- `src/augmented_dataset.py:49`
- `src/augmented_dataset.py:50`
- `src/augmented_dataset.py:51`

本项目没有 whole-cell stain。为了尽量对齐 CellSAM 的 `(R=blank, G=nuclear, B=whole-cell)`，T31 采用以下近似映射：

- `R = 0`
- `G = DAPI`
- `B = BF`

即：

```text
Cellpose input RGB = [blank, DAPI, BF]
channels = [3, 2]
```

说明：
- `DAPI` 对应 nuclear
- `BF` 是当前最接近 whole-cell 结构线索的通道
- 这不是完美等价于 CellSAM 训练集中的 whole-cell stain，而是当前数据条件下最接近论文口径的实现

## 4. 最小实现清单

### 4.1 新建独立脚本

不要继续扩展 `tools/baseline_eval.py`。

新建：

- `tools/cellpose_paper_aligned_eval.py`

理由：

1. `tools/baseline_eval.py` 已废弃，且混有旧 CellSAM baseline 兼容逻辑。
2. T31 需要单独保存方法学口径、输入映射、Cellpose 参数和逐样本结果。
3. T31 是论文基线修正，不应继续依赖历史脚本。

### 4.2 数据输入

对每张 `test(73)` 图像：

1. 读取原始 3 通道数组
2. 取：
   - `bf = raw_img[0]`
   - `dapi = raw_img[1]`
3. 逐通道归一化到 `[0,1]`
4. 组装为：

```python
rgb = np.stack([zeros, dapi, bf], axis=-1).astype(np.float32)
```

### 4.3 模型调用

主实验固定为：

- `model_type='cyto3'`
- `channels=[3,2]`
- `diameter=None`

建议实现：

```python
from cellpose import models

model = models.Cellpose(model_type="cyto3", gpu=True)
masks, flows, styles, diams = model.eval(
    rgb,
    channels=[3, 2],
    diameter=None,
)
```

关键约束：

1. 必须显式指定 `cyto3`
2. 必须显式指定 `channels=[3,2]`
3. 不再使用“默认构造 + BF 灰度”的旧路径

### 4.4 指标输出

每次运行同时输出两套指标。

#### A. 当前项目指标

- `PQ`
- `BM-1to1 Dice`
- `BM-Coverage Dice`
- `AJI`
- `Semantic Dice`
- `TP/FP/FN`

理由：
- 便于和现有 baseline 表直接对齐

#### B. CellSAM 论文口径指标

- `F1`
- `Recall`

实现参考：
- `cellSAM_source/paper_evaluation/cpm.py`

理由：
- 便于回答“与 CellSAM 论文里的 Cellpose 是否同方向”

### 4.5 输出文件

建议输出到：

- `experiments/cellpose_paper_aligned_test73/results.json`
- `experiments/cellpose_paper_aligned_test73/per_sample_cellpose_cyto3_gdapi_bbf.json`

其中 `results.json` 至少记录：

1. `split`
2. `n_samples`
3. `model_type`
4. `channels`
5. `input_encoding`
6. `diameter`
7. 项目指标均值和标准差
8. `F1`
9. `Recall`
10. 执行时间
11. 脚本名

## 5. 扩展实验

### 5.1 主结果

主结果只保留一行：

- `Cellpose cyto3, [0,DAPI,BF], channels=[3,2], diameter=None`

### 5.2 补充敏感性实验

主结果跑完后，只做一个补充：

- `diameter=200`

如果仍需更严格，可在 `val(71)` 做小扫描：

- `diameter ∈ {120, 160, 200, 240}`

然后只把锁定后的最优直径带到 `test(73)` 跑一次。

约束：
- 不允许用 `test(73)` 反向调参

## 6. 解释框架

### 情况 A

`PQ/F1` 仍接近 0

解释：
1. Cellpose 在当前心肌细胞任务上确实不适用
2. 历史 baseline 虽然方法学错配，但主结论方向不变

### 情况 B

结果明显高于历史 BF-only baseline，但仍远低于 Ours / MedSAM / CellSAM-official

解释：
1. 历史 baseline 的确低估了 Cellpose
2. 但 Cellpose 仍不是强基线

### 情况 C

结果接近中强 baseline

解释：
1. 历史 baseline 不可再引用
2. 必须用 T31 替换论文中的 Cellpose 行

## 7. 执行顺序

1. 新建 `tools/cellpose_paper_aligned_eval.py`
2. 在 `test(73)` 跑主结果：
   - `cyto3`
   - `[0,DAPI,BF]`
   - `channels=[3,2]`
   - `diameter=None`
3. 输出项目指标和 `F1/Recall`
4. 如主结果仍较差，再补 `diameter=200`
5. 回填：
   - `docs/experiments_log.md`
   - 论文结果表
   - `docs/agent_inbox.md`

## 8. 完成标准

- [ ] 独立脚本可运行
- [ ] `test(73)` 主结果生成
- [ ] 逐样本 JSON 生成
- [ ] 项目指标和 `F1/Recall` 同时输出
- [ ] 与历史 BF-only baseline 的方法学差异写清
- [ ] 明确哪一版可进入论文

## 9. 当前结论

在 T31 完成前：

1. `tools/baseline_eval.py` 里的 Cellpose 结果不应用于正式论文结论
2. `diameter=200` 的历史补充只能算旧实现下的辅助观察
3. 当前最稳妥的口径是：**Cellpose baseline 需要按 CellSAM 论文口径重跑**
