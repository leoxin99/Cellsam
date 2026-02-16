---
description: CellSAM 项目常用命令 - 自动执行
---

# CellSAM 项目自动化工作流

## 命令安全级别说明

### ✅ 安全命令 (自动执行)
这些命令不会修改代码、数据或实验结果，可以安全地自动执行。

// turbo-all

### ⚠️ 需审批命令 (需要用户确认)
这些命令可能修改重要文件或执行不可逆操作，需要用户确认。

---

## 安全命令列表 (// turbo)

### 1. SSH 连接检查
// turbo
```bash
ssh alice "hostname; whoami"
```

### 2. 查看作业状态
// turbo
```bash
ssh alice "squeue -u s3890074"
```

### 3. 查看训练日志
// turbo
```bash
ssh alice "tail -100 ~/CellSam/logs/p2a_*.log"
ssh alice "ls -lt ~/CellSam/logs/*.log | head -5"
```

### 4. 检查文件/目录
// turbo
```bash
ssh alice "ls -la ~/CellSam/data/processed/images/ | head -20"
ssh alice "ls ~/CellSam/data/processed/images/ | wc -l"
```

### 5. 验证环境
// turbo
```bash
ssh alice "bash -l -c 'eval \"\$(conda shell.bash hook)\"; conda activate cellsam; python --version'"
```

### 6. 运行回归测试
// turbo
```bash
conda run -n cellsam python tools/test_unified_regression.py
```

### 7. 运行梯度门禁
// turbo
```bash
conda run -n cellsam python tools/test_loss_gradients.py
```

### 8. 本地文件查看
// turbo
```powershell
Get-ChildItem -Path "path" | Measure-Object
Get-Content file.txt | Select-Object -First 10
```

---

## 需审批命令列表

### ⚠️ 训练提交
```bash
ssh alice "sbatch scripts/train_phase2a.sh"
ssh alice "sbatch scripts/train_phase2a_a100.sh"
```

### ⚠️ 评估运行
```bash
conda run -n cellsam python tools/standardized_inference.py --checkpoint <path>
conda run -n cellsam python tools/evaluate_e2e.py
conda run -n cellsam python tools/comprehensive_eval.py
```

### ⚠️ 检测消融
```bash
conda run -n cellsam python tools/ablation_detection_e34b.py
conda run -n cellsam python tools/ablation_detection_lock.py
```

### ⚠️ 代码修改
所有对 `src/`, `tools/`, `scripts/` 的编辑

### ⚠️ 数据上传/删除
```bash
scp -r data/ alice:~/CellSam/
ssh alice "rm -rf ~/CellSam/data/processed/*"
```

### ⚠️ 环境修改
```bash
ssh alice "conda install package"
ssh alice "pip install package"
```

### ⚠️ 配置修改
`src/config/*.yaml` 文件的编辑

---

## 安全建议

| 命令类型 | 安全级别 | 原因 |
|----------|----------|------|
| 读取/查看 | ✅ 安全 | 只读操作 |
| 状态检查 | ✅ 安全 | 不修改任何内容 |
| 日志查看 | ✅ 安全 | 只读操作 |
| 回归测试 | ✅ 安全 | 只读检查 |
| 代码编辑 | ⚠️ 审批 | 影响项目行为 |
| 训练提交 | ⚠️ 审批 | 消耗计算资源 |
| 文件删除 | ⚠️ 审批 | 不可逆操作 |
| 环境安装 | ⚠️ 审批 | 可能破坏环境 |
