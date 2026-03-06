# DAPI Only 检测方案完整设计文档

> **维护者**: Research Documentation Architect
> **创建日期**: 2026-01-31
> **最后更新**: 2026-03-06
> **代码位置**: `src/detection/dapi.py`
> **当前 SSOT**: 统一评估/封板参数以 `src/detection/profiles.py` 的 `locked_eval` 为准；`src/detection/dapi.py` 仅代表运行时默认值

---

## 一、方案概述

DAPI Only 方案通过 DAPI 核染色通道检测细胞核，为 SAM 分割模型提供边界框 prompt。

> 当前项目存在两套参数口径:
> 1. `runtime default`: `src/detection/dapi.py` 函数签名中的默认值，用于日常运行与开发
> 2. `locked_eval`: `src/detection/profiles.py` 中的冻结参数，用于统一评估、阶段结论和 test 封板

### 核心流程

```
DAPI 图像 (1024×1024)
        ↓
┌───────────────────┐
│   detect_nuclei   │  Otsu 阈值 + 形态学
└────────┬──────────┘
         ↓
    核区域列表
         ↓
┌───────────────────┐
│ merge_close_nuclei│  并查集合并双核
└────────┬──────────┘
         ↓
    细胞分组
         ↓
┌───────────────────┐
│create_bounding_boxes│ 各向异性扩展
└────────┬──────────┘
         ↓
    边界框 [[x1,y1,x2,y2], ...]
```

---

## 二、版本演进历史

### v1.0 (E03, 2026-01-08) - 初始版本

**来源**: E02 CellFinder 失败后，开发 DAPI 检测替代方案

| 参数 | 初始值 | 说明 |
|------|--------|------|
| min_nucleus_area | **500** | 最小核面积 |
| max_nucleus_area | **15000** | 最大核面积 |
| merge_distance | **100px** | 固定合并距离 |
| edge_margin | **30px** | 边缘排除 |

**结果**: F1 = **0.750** (vs CellFinder 0.012)

**代码**: `anti_test/test_dapi_detection.py` (原始测试脚本)

---

### v1.1 (E04-E05, 2026-01-09) - 管线集成

**改动**:
- 将 DAPI 检测集成到 SAM 分割管线
- 添加实例级 mask 输出 (非像素级合并)

**结果**: Overall Dice 0.58 → **0.71**

---

### v1.2 (E06, 2026-01-11) - 分水岭尝试 (失败)

**尝试**:
- 使用分水岭算法分离粘连核
- 添加 circularity 过滤

**结果**: F1 **0.34** (严重下降 -0.41)

**原因**: 心肌细胞核形态不规则，分水岭导致过度分割

**决策**: ❌ 放弃分水岭，保持简单 Otsu 方案

---

### v1.3 (E14, 2026-01-14) - 各向异性扩展

**背景**: 发现核-细胞轴向对齐率仅 50% @ 30°

**改动**:
- 添加各向异性扩展 (沿核长轴方向扩展更多)
- 引入 `expansion_long` = 5.0, `expansion_short` = 3.0

```python
if aspect < 1.3:  # 圆形核
    expand = 4.0  # 各向同性
else:  # 椭圆核
    expand_long = 5.0   # 长轴
    expand_short = 3.0  # 短轴
```

**结果**: Dice +3.3%

---

### v2.0 (E18, 2026-01-23) - 添加心肌细胞过滤

**新增函数**:
- `filter_by_actn2()`: 使用 Actn2 覆盖率过滤非心肌细胞
- `detect_cardiomyocytes()`: 完整 DAPI+Actn2 检测流程

**参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| coverage_threshold | 0.3 | 最小 Actn2 覆盖率 |
| intensity_threshold | 0.2 | 最小 Actn2 强度 |

---

### v3.0 (E18, 2026-01-23) - Adaptive 方案

**新增函数**:
- `detect_with_adaptive_box()`: Z-线引导自适应框
- `detect_zlines_in_region()`: blob_log Z-线检测
- `create_adaptive_box()`: 自适应框生成

**参数** (更新 2026-02-05):
| 参数 | 旧值 | 新值 (1024px) | 说明 |
|------|------|---------------|------|
| search_radius | 400 | **256** | Z-线搜索半径 |
| min_zlines | 15 | **15** | 最小 Z-线数 |
| zline_threshold | 0.03 | **0.03** | blob_log 阈值 |

---

### v4.0 (E19, 2026-01-26) - 参数精确化

**背景**: E18 发现边缘过滤过松、双核合并阈值不准

**数据驱动分析**:
1. GT 极小核分析: \<1000px 为碎片，有效核 \>3000px
2. 边缘排除率统计: 100px 排除 5.6% 边缘核
3. 双核间距统计: Mean=161px, P75=160px

**参数更新**:
| 参数 | 旧值 | 新值 | 依据 |
|------|------|------|------|
| min_nucleus_area | 500 | **3000** | GT 分析 |
| edge_margin | 30 | **100** | 边缘排除率 |
| merge_distance | 100 固定 | **1.2×直径** | 双核间距分布 |

---

### v4.1 (E22, 2026-01-30) - 消融验证

**实验**: 参数消融实验 (20 个 test 样本)

| 参数 | 最佳值 | F1 |
|------|--------|-----|
| min_nucleus_area | **2000** | **0.806** |
| max_nucleus_area | 15000+ | 0.765 |
| merge_distance | relative | 0.765 |

**发现**: min_area=2000 优于 3000，提升 4%

---

### v5.0 (2026-02-05) - 分辨率修正 ⭐

**背景**: 发现原始图像分辨率记录错误 (Error 7)

**问题**:
- 文档记录图像尺寸: 1608×1608 (错误)
- 实际图像尺寸: **1736×1776** (正确)
- 训练分辨率: 1024×1024
- 缩放系数: (1024/1756)² = **0.340** (非 0.4055)

**参数重新统计** (Dev Set 50张, 1024px):
| 统计量 | 值 | 说明 |
|--------|-----|------|
| 核 P1 | 57 | 最小阈值参考 |
| 核 Median | 3268 | 典型核大小 |
| 核 P99 | 10026 | 最大阈值参考 |
| 边距 P5 | 6 | margin 参考 |

**参数更新** (全部函数):
| 参数 | 旧值 | 新值 | 函数 |
|------|------|------|------|
| min_area | 500 | **200** | detect_nuclei |
| max_area | 30000 | **10000** | detect_nuclei |
| margin | 30/50 | **20/32** | is_on_edge, create_bounding_boxes |
| search_radius | 400 | **256** | detect_zlines_in_region |

---

## 三、当前默认运行参数 (更新 2026-02-05)

> 说明: 本节是代码默认值 (runtime default)；用于统一评测/封板的参数以 **3.1 锁定进展** 为准。

| 参数 | 值 | 分辨率 | 来源 |
|------|-----|--------|------|
| min_nucleus_area | **200** | 1024px | 2026-02-05 重新统计 |
| max_nucleus_area | **10000** | 1024px | P99 @ 1024 |
| use_relative_distance | **True** | - | E19/E22 |
| merge_coeff | 1.2 | - | E19 数据驱动 |
| size_ratio_threshold | 3.0 | - | 经验值 |
| edge_margin | **32** | 1024px | scaled from 50 |
| expansion_long | 5.0 | - | E14 分析 |
| expansion_short | 3.0 | - | E14 分析 |
| search_radius | **256** | 1024px | ~P99 box/2 |
| round_threshold | 1.3 | - | 经验值 |

---

### 3.1 锁定进展更新 (2026-02-14)

> 为避免 test 泄漏，检测参数采用 `val(71) 调参 -> test(73) 单次封板` 协议。
> 状态: ✅ 当前项目统一评估以 `locked_eval` 为准；test73 结果保留为冻结评估记录，不再反向改写参数。

| 项目 | 当前状态 | 参数 | F1 | 说明 |
|------|----------|------|----|------|
| DAPI 默认运行参数 | 保留 | `min=200`, `max=10000`, `relative=True` | - | 运行时默认 (`src/detection/dapi.py`) |
| DAPI `locked_eval` | 当前生效 | `min=1500`, `max=20000`, `edge=20`, `ratio=2.5`, `merge=1.4`, `relative_1.2x` | `0.8106` (val71) | 统一评估口径 (`src/detection/profiles.py`) |
| Adaptive `locked_eval` | 当前生效 | `radius=160`, `min_zlines=5`, `zline_threshold=0.05`, `edge=20`, `ratio=2.5`, `merge=1.4` | `0.7800` (val71) | 统一评估口径 (`src/detection/profiles.py`) |
| test 封板记录 | 历史冻结 | 固定 `locked_eval` 候选后单次评估 | `0.8033` (DAPI) | DAPI > Adaptive，保留追溯 (`experiments/ablation_detection_lock/results.json`) |

> T4 更新 (2026-02-16): 检测评估脚本已接入 profile 防呆机制；当前仅 `locked_eval` 作为活跃 profile，运行时默认值仅保留在 `dapi.py` 代码签名中供开发参考。实现见 `src/detection/profiles.py`。
> T3b 更新 (2026-02-19): 半径重扫后 Adaptive 在 `val(71)` 上提升到 F1=0.7800，但该结果属于封板后诊断回合，`test(73)` 锁定结果不回写。

### 3.2 Adaptive 退化诊断补充 (T3, 2026-02-16)

> 目标: 判断 B2/B3 不敏感是“参数本身不敏感”还是“fallback 掩盖差异”。

- 数据与口径:
  - 数据集: `val(71)`
  - 结果文件: `experiments/ablation_adaptive_val/results.json`
  - 诊断快照: `experiments/ablation_adaptive_val/diagnosis_t3.json`
- 关键观测:
  - B2 (`min_zlines`) F1 区间: `0.7472 -> 0.7472`，range=`0.0000`
  - B3 (`zline_threshold`) F1 区间: `0.7459 -> 0.7472`，range=`0.0013`
  - `adaptive_ratio=1.0`, `fallback_count=0`（所有组合均无 fallback）
  - `mean_zlines` 仍较高（B2 均值约 `1425.4`，B3 区间约 `1070.0-1571.2`）
- 诊断结论:
  - `cause_code = zline_saturated`
  - 当前 `search_radius=200` 下，Adaptive 始终走自适应分支，B2/B3 的阈值变化无法有效改变框生成，故表现为“近似平坦”。

---

## 四、核心算法细节

### 4.1 detect_nuclei()

```python\ndef detect_nuclei(dapi_channel, min_area=200, max_area=10000):  # Updated 2026-02-05
    # 1. 对比度归一化 (P2-P98 拉伸)
    p2, p98 = np.percentile(dapi_channel, [2, 98])
    img_norm = (dapi_channel - p2) / (p98 - p2)
    
    # 2. Otsu 阈值分割
    thresh = threshold_otsu(img_norm)
    binary = img_norm > thresh
    
    # 3. 形态学清理
    binary = binary_opening(binary, disk(3))  # 去除小噪点
    binary = binary_fill_holes(binary)  # 填充内部空洞
    
    # 4. 连通域标记 + 面积过滤
    labels = label(binary)
    regions = [r for r in regionprops(labels) 
               if min_area <= r.area <= max_area]
    
    return regions
```

### 4.2 merge_close_nuclei()

```python
def merge_close_nuclei(regions, size_ratio_threshold=3.0, use_relative_distance=True):
    # 1. 计算核直径
    diameters = [2 * sqrt(r.area / π) for r in regions]
    
    # 2. 并查集合并
    for i, j in all_pairs:
        # 尺寸相似性检查
        if max(d[i], d[j]) / min(d[i], d[j]) > 3.0:
            continue
        
        # 距离阈值
        if use_relative_distance:
            max_dist = 1.2 * (d[i] + d[j]) / 2
        else:
            max_dist = 373  # 旧固定值
        
        if distance[i,j] < max_dist:
            union(i, j)
    
    return grouped_regions
```

### 4.3 create_bounding_boxes()

```python
def create_bounding_boxes(cell_groups, image_shape, ...):
    for group in cell_groups:
        # 1. 边缘排除
        if any(is_on_edge(r, margin=32) for r in group):  # Updated 2026-02-05
            continue
        
        # 2. 计算核长宽比
        aspect = max(nuc_w, nuc_h) / min(nuc_w, nuc_h)
        
        # 3. 选择扩展策略
        if aspect < 1.3:  # 圆形
            expand_x = expand_y = 4.0
        else:  # 椭圆
            expand_long, expand_short = 5.0, 3.0
        
        # 4. 生成边界框
        box_w = nuc_w * expand_x
        box_h = nuc_h * expand_y
        boxes.append([cx-box_w/2, cy-box_h/2, cx+box_w/2, cy+box_h/2])
```

---

## 五、评估指标

| 指标 | 公式 | 用途 |
|------|------|------|
| Precision | TP / (TP + FP) | 检测准确性 |
| Recall | TP / (TP + FN) | 检测覆盖率 |
| F1 | 2×P×R / (P+R) | 综合指标 |
| IoU 阈值 | 0.3 | 框匹配阈值 |

---

## 六、实验记录溯源

| 实验 | 日期 | 改动 | 数据集 | 结果文件 |
|------|------|------|--------|----------|
| E03 | 01-08 | 初始版本 | ~50 随机 | ❌ 无 |
| E06 | 01-11 | 分水岭 (失败) | - | ❌ 无 |
| E14 | 01-14 | 各向异性扩展 | - | ❌ 无 |
| E18 | 01-23 | SarcGraph 对比 | 5 test | ✅ 有 (PNG) |
| E19 | 01-26 | 边缘/双核参数 | - | ❌ 无 |
| E20 | 01-30 | DAPI vs Adaptive | 20 test | ✅ 有 (JSON) |
| E22 | 01-30 | 参数消融 | 20 test | ✅ 有 (JSON) |
| E34b | 02-14 | 边缘/双核联合消融 (`edge_margin/size_ratio/merge_coeff`) | 71 val | ✅ `experiments/ablation_detection_e34b/results.json` |
| E34-lock | 02-14 | DAPI vs Adaptive 单次封板 | 73 test | ✅ `experiments/ablation_detection_lock/results.json` |

⚠️ **注意**: E03, E06, E14, E19 没有保存 results.json，参数来源为 experiments_log.md 和代码注释。

---

## 七、待改进

1. **早期实验无结果文件**: E03 参数来源不可追溯（历史遗留）
2. **Adaptive 方案需迭代**: 当前 test73 F1=0.7502，落后 DAPI 的 0.8033
3. **参数治理需持续**: 保持 runtime default 与 locked eval 的双轨并避免混用

---

## 八、章节更新方案 (2026-02-14)

> 目标: 解决“默认参数 / 锁定候选 / 最终封板”混写问题，避免新对话误读。

| 章节 | 当前问题 | 更新动作 | 优先级 |
|------|----------|----------|--------|
| 一、方案概述 | 默认运行与评测锁定区分不够显式 | 增加“运行默认值 vs 评测锁定值”提示，并指向 3.1 | High |
| 二、版本演进历史 | 缺 E34 的 val 锁定与 test 封板节点 | 新增 v5.1 (E34-val) 与 v5.2 (E34-test lock) | High |
| 三、参数 | 已拆默认与锁定，但需强调当前只以 `locked_eval` 为活跃口径 | 增加“code default vs locked_eval vs historical test record”优先级说明 | High |
| 四、核心算法细节 | DAPI 与 Adaptive 共用参数面未集中说明 | 新增“共用参数表”: `min/max_nucleus_area`, `size_ratio_threshold`, `use_relative_distance`, `edge_margin` | High |
| 五、评估指标 | 缺 E34 统一口径说明 | 明确 IoU=0.3, micro P/R/F1, val 调参 + test 单次锁定 | Medium |
| 六、实验记录溯源 | 缺最新结果文件挂载 | 增补 `ablation_dapi_val` 与 `ablation_adaptive_val` 的结果路径 | High |
| 七、待改进 | 条目偏旧，未覆盖当前已知风险 | 增加 E34b: `edge_margin`, `size_ratio_threshold`, `merge_coeff` 联合消融与 Adaptive 退化诊断 | High |

### 8.1 术语澄清 (针对极小核争议)

- “GT 极小核/碎片”是**核级统计口径**下的可疑目标，不能等价为“必须删除 GT 实例”。
- 当前策略:
  - 训练/实例评估中不对 GT 实例做静默面积过滤；
  - 在“核检测参数推导”(边缘过滤、双核合并统计)中，可使用“有效核”定义，但必须显式标注分辨率与数据集口径。
