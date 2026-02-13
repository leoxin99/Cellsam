# ALICE 训练快速参考 (永久文档)

> **最后更新**: 2026-02-13

---

## 一、登录信息

| 项目 | 值 |
|------|-----|
| **集群地址** | `login.alice.universiteitleiden.nl` |
| **ULCN 用户名** | `s3890074` |
| **SSH 命令** | `ssh s3890074@login.alice.universiteitleiden.nl` |
| **VPN** | EduVPN → Leiden University → ULCN 登录 |
| **项目目录** | `~/CellSam` |
| **Conda 环境** | `cellsam` |

---

## 二、GPU 分区 (sinfo 查询 2026-02-10)

| 分区 | GPU | 显存 | 节点数 | GPU/节点 | 时间限制 | 推荐场景 |
|------|-----|------|--------|----------|----------|----------|
| `gpu-short` | 2080Ti/A100/L4 | 混合 | ~22 | 2-4 | 4h | 测试调试 |
| `gpu-l4-24g` | L4 | 24GB | 7 | 4 | 7天 | 常规训练 |
| **`gpu-a100-80g`** | **A100** | **80GB** | **6** | **2** | **7天** | **⭐ 推荐** |

> ⚠️ `gpu-a100` (40G) 分区已不存在。旧脚本中的 `--partition=gpu-a100` 需改为 `gpu-a100-80g`。

> ⚠️ **CUDA Module 更新 (2026-02-13)**: `cuda/11.8` 已移除，当前可用: `CUDA/12.1.1`, `CUDA/12.3.0`。
> SLURM 脚本统一使用 `module load CUDA/12.1.1`。
> Login 节点用 conda 自带 CUDA 可跑测试，但 compute 节点必须正确加载 module。

---

## 三、常用命令

```bash
# 登录
ssh <用户名>@login.alice.universiteitleiden.nl

# 激活环境
conda activate cellsam

# 提交任务
sbatch scripts/<script_name>.sh

# 查看任务
squeue -u $USER

# 实时日志
tail -f logs/<job_name>_<job_id>.log

# 取消任务
scancel <job_id>

# 维护后检查（重要！）
module avail CUDA
conda info --base
```

---

## 四、CellSAM 训练环境配置 (一次性)

```bash
# 注意: Miniforge3 版本号会随系统维护变化，用 module avail Miniforge3 查看当前版本
module load Miniforge3/24.11.3-0
conda create --name cellsam python=3.11 -y
conda activate cellsam
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install segment-anything scikit-image scikit-learn albumentations dask tqdm
```

---

## 五、SLURM 脚本模板 (A100)

```bash
#!/bin/bash -l
#SBATCH --job-name=cellsam_train
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -eo pipefail                        # 不加 -u，等 conda 初始化完再加

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u                                  # conda 激活完毕，启用 nounset

export PYTHONPATH=$PYTHONPATH:~/CellSam/cellSAM_source
cd ~/CellSam
mkdir -p logs checkpoints

python src/train.py --config src/config/CONFIG_FILE.yaml
```

---

## 六、Git 管理设置 (推荐)

> **之前使用 scp 上传，现改为 git 管理更可靠**

### 首次设置 (在 ALICE 上一次性执行)

```bash
# SSH 登录后
cd ~/CellSam

# 如果已有非 git 文件，先备份 checkpoints
mv checkpoints ~/checkpoints_backup

# 初始化 git
rm -rf .git  # 清理可能存在的不完整 .git
git init
git remote add origin https://github.com/leoxin99/CellSam.git
git fetch origin
git reset --hard origin/main

# 恢复 checkpoints
mv ~/checkpoints_backup checkpoints
```

### 日常更新流程

```bash
# 本地修改后
git add -A && git commit -m "Update" && git push

# ALICE 上同步
ssh s3890074@login.alice.universiteitleiden.nl "cd ~/CellSam && git pull origin main"
```

### SCP vs Git 对比

| 方面 | SCP | Git |
|------|-----|-----|
| **可靠性** | ⚠️ 易遗漏 | ✅ 自动同步 |
| **版本追踪** | ❌ 无 | ✅ 有 |
| **推荐场景** | 紧急修复 | 日常开发 |

---

## 七、系统维护踩坑记录

> Alice 集群会定期维护，可能导致环境变化。以下是已遇到的问题。

| 日期 | 变化 | 影响 | 修复 |
|------|------|------|------|
| ~2026-02 | `cuda/11.8` 移除 | SLURM `module load` 失败 | 改用 `CUDA/12.1.1` |
| ~2026-02 | Miniforge3 升级，`~/miniconda3` 消失 | `source ~/miniconda3/conda.sh` 失败 | 改用 `eval "$(conda shell.bash hook)"` |
| ~2026-02 | MKL 激活脚本引用未定义变量 | `set -u` 导致脚本静默退出 | `set -u` 移到 conda activate 之后 |

**教训**:
1. **Login ≠ Compute**: login 节点能跑通的命令不代表 SLURM 脚本也能跑通（初始化路径不同）
2. **维护后必检**: 每次 Alice 维护后先检查 `module avail CUDA` 和 `conda info --base`
3. **不硬编码路径**: 用 `eval "$(conda shell.bash hook)"` 替代 `source ~/miniconda3/...`
4. **注意 set -u**: conda 的激活脚本会引用未定义变量，`set -u` 必须在 conda activate 之后
