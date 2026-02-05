# DAPI Only 检测方案完整设计文档

> **维护者**: Research Documentation Architect
> **创建日期**: 2026-01-31
> **代码位置**: `src/detection/dapi.py`

---

## 一、方案概述

DAPI Only 方案通过 DAPI 核染色通道检测细胞核，为 SAM 分割模型提供边界框 prompt。

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

## 三、当前最优参数 (更新 2026-02-05)

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

⚠️ **注意**: E03, E06, E14, E19 没有保存 results.json，参数来源为 experiments_log.md 和代码注释。

---

## 七、待改进

1. **数据集不一致**: 消融实验应使用 val 集 (71样本)，而非 test 集
2. **早期实验无结果文件**: E03 参数来源不可追溯
3. **Adaptive 方案需迭代**: search_radius=200 后 F1=0.755，接近 DAPI Only
