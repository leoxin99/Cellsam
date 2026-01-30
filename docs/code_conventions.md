# CellSAM 代码规范与标准函数

> **目的**: 防止新代码与已有方案不一致的问题
> **创建**: 2026-01-23
> **强制要求**: 编写新代码前必须查阅此文档

---

## 1. 检测模块标准入口

### ⭐ DAPI 核检测 (正确方法)

**用途**: 检测心肌细胞并生成框

**标准调用**:
```python
from detection.dapi import detect_cardiomyocytes

# 正确方法: DAPI 定位 + Actn2 过滤
boxes, filtered_groups, all_groups, regions = detect_cardiomyocytes(
    dapi_channel=img[4],      # Ch4 = DAPI
    actn2_channel=img[1],     # Ch1 = Actn2
    min_nucleus_area=500,
    max_nucleus_area=30000,
    actn2_coverage_threshold=0.3
)
```

**❌ 错误用法** (不要这样做):
```python
# 错误: 只用 DAPI，没有 Actn2 过滤
regions = detect_nuclei(dapi)
groups = merge_close_nuclei(regions)
boxes = create_bounding_boxes(groups, dapi.shape)
```

---

### ⭐ Z-线引导自适应框

**用途**: 根据 Z-线范围确定框大小

**标准调用**:
```python
from detection.dapi import detect_with_adaptive_box

boxes, cell_groups, debug_info = detect_with_adaptive_box(
    dapi_channel=img[4],
    actn2_channel=img[1],
    search_radius=400,      # 待优化
    min_zlines=15,
    zline_threshold=0.03,
    exclude_edges=True,
    margin=30
)
```

---

## 2. 数据通道映射

| 变量名 | 通道 | 用途 |
|--------|------|------|
| `bf` | img[0] | Brightfield |
| `actn2` | img[1] | α-actinin (Z-线) |
| `dapi` | img[4] | 细胞核 |
| `gt` | img[9] | GT 实例掩膜 |

**⚠️ 警告**: Ch5 在部分文档标记为 Actn2，但实际为空！

---

## 3. GT 框提取

**标准方法**:
```python
from skimage import measure

def get_gt_boxes(gt_mask):
    """提取所有 GT 细胞的边界框 (不过滤边缘)"""
    boxes = []
    for region in measure.regionprops(gt_mask.astype(int)):
        y1, x1, y2, x2 = region.bbox
        boxes.append([x1, y1, x2, y2])
    return boxes
```

---

## 4. 数据划分 (2026-01-25)

**文件**: `data/splits/split_v1.json`

| 集合 | 用途 | 数量 |
|------|------|------|
| Dev | 参数推导 | 50 张 |
| Train | 训练 (含 Dev) | 400 张 |
| Test | 独立评估 | 78 张 |

---

## 5. 边缘过滤规则 (更新)

**推荐阈值**: 100-150 px (核心距边缘)

```python
# 排除距离边缘 < 100px 的核
edge_dist = min(cy, h-cy, cx, w-cx)
if edge_dist < 100:
    exclude
```

---

## 6. 双核合并参数 (更新)

**推荐阈值**: 373 px (Dev Set P95)

```python
# 默认使用固定阈值 373px
merge_close_nuclei(regions, use_relative_distance=False, fixed_merge_distance=373)
```

---

## 7. 新代码自检清单

写新代码前必须确认:

- [ ] 是否已查阅此文档的标准函数？
- [ ] 是否使用了正确的通道映射？
- [ ] 是否复用了已有函数，而非重新实现？
- [ ] 参数是否来自 `dataset_parameters.md`？
- [ ] 是否与 `anti_test/` 中已验证的代码一致？

---

## 更新日志

| 日期 | 更新 |
|------|------|
| 2026-01-23 | 初始创建，定义标准检测函数入口 |
