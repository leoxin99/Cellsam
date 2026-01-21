# CellSAM 常见问题与解决方案

> **文档类型**: 故障排除指南 (从 CLAUDE.md 提取)
> **最后更新**: 2026-01-21

---

## Q1: TIFF 读取慢

**症状**: 训练数据加载非常慢

**解决**: 转换为 NPY 格式，读取速度提升 100x

```bash
python data/scripts/extract_expanded_pairs.py --limit 50
```

---

## Q2: Dice 卡住不上升

**症状**: 验证 Dice 始终为 0.0000，但 Loss 正常下降

**原因**: 类别不平衡导致模型预测全背景

**解决**:
1. 检查 GT 框质量 (前景 > 10%)
2. 降低学习率至 5e-5
3. 减弱数据增强强度
4. 确保使用边界框内损失计算

---

## Q3: GPU 内存不足

**症状**: CUDA out of memory

**解决**:
- 减小 batch_size 至 2
- 减小图像尺寸至 512
- 使用 AMP 混合精度训练 (`use_amp: true`)

---

## Q4: CellFinder 漏检

**症状**: CellFinder 检测 F1 = 0.012 (极低)

**原因**: CellFinder 主要针对小型圆形细胞训练，心肌细胞太大且不规则

**解决**:
- 使用 DAPI 核检测替代 (F1 = 0.750)
- 智能双核合并处理双核细胞

---

## Q5: 边界锯齿

**症状**: 分割边界有锯齿状突起

**解决**: 
使用 6 步边界平滑管道:
1. Morphological closing
2. Fill holes
3. Gaussian smoothing (σ=7)
4. Binary opening (disk=8)
5. Binary closing (disk=8)
6. Second Gaussian pass (σ=5)

代码位置: `src/inference/postprocess.py`

---

## Q6: 相邻细胞同色

**症状**: 可视化时相邻细胞颜色相同，难以区分

**解决**:
使用图着色算法 (4-color theorem):
```python
from inference import mask_to_rgb
rgb = mask_to_rgb(instance_mask)  # 相邻细胞保证不同色
```

代码位置: `src/inference/visualize.py`

---

## Q7: 细胞过大/过小

**症状**: 预测的细胞面积异常

**解决**:
应用大小过滤:
- MIN_CELL_AREA = 40,836 像素 (P1)
- MAX_CELL_AREA = 513,928 像素 (P99)

代码位置: `src/inference/postprocess.py`

---

*更多问题请联系项目维护者*
