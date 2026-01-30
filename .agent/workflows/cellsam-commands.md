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

### 3. 查看日志
// turbo
```bash
ssh alice "tail -100 ~/CellSam/logs/cellsam_*.log"
ssh alice "cat ~/CellSam/logs/cellsam_*.err"
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
ssh alice "bash -l -c 'conda activate cellsam; python --version'"
ssh alice "bash -l -c 'conda activate cellsam; python ~/CellSam/verify_env.py'"
```

### 6. 本地文件查看
// turbo
```bash
Get-ChildItem -Path "path" | Measure-Object
Get-Content file.txt | Select-Object -First 10
```

---

## 需审批命令列表

### ⚠️ 数据处理
```bash
python data/scripts/extract_expanded_pairs.py  # 修改 processed/ 目录
```

### ⚠️ 训练提交
```bash
ssh alice "sbatch scripts/train_semantic.sh"  # 消耗 GPU 资源
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
ssh alice "conda remove --name cellsam --all"
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
| 代码编辑 | ⚠️ 审批 | 影响项目行为 |
| 数据处理 | ⚠️ 审批 | 覆盖现有数据 |
| 训练提交 | ⚠️ 审批 | 消耗计算资源 |
| 文件删除 | ⚠️ 审批 | 不可逆操作 |
| 环境安装 | ⚠️ 审批 | 可能破坏环境 |

---

## 使用方法

在对话中使用 `/cellsam-commands` 触发此工作流。

标记为 `// turbo` 的命令将自动执行，无需用户确认。
