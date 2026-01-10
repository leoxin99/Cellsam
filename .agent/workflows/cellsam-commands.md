---
description: CellSAM 项目常用命令 - 自动执行
---

# CellSAM 项目快捷工作流

本工作流包含项目中常用的命令，使用 `// turbo` 标注的命令会自动执行。

## 1. 激活环境
// turbo
```powershell
conda activate cellsam
```

## 2. 运行训练
// turbo
```powershell
conda run -n cellsam python train_expanded.py --epochs 20 --batch_size 2
```

## 3. 测试模型
// turbo
```powershell
conda run -n cellsam python test_model.py
```

## 4. 下载 Allen 数据
// turbo
```powershell
conda run -n cellsam python download_full_segmented.py
```

## 5. 提取训练对
// turbo
```powershell
conda run -n cellsam python extract_expanded_pairs.py
```

## 6. 分析数据结构
// turbo
```powershell
conda run -n cellsam python analyze_segmented_data.py
```

## 7. 查看 Napari 可视化
```powershell
conda run -n cellsam python view_test_results.py
```

## 8. 检查 GPU 状态
// turbo
```powershell
nvidia-smi
```

## 9. 安装 Python 包
// turbo
```powershell
conda run -n cellsam pip install [package_name] -q
```

## 10. 列出目录内容
// turbo
```powershell
Get-ChildItem -Path [path] | Format-Table Name, Length -AutoSize
```
