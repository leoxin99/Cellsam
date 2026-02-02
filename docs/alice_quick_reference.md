# ALICE 训练快速参考 (永久文档)

> **最后更新**: 2026-02-02

---

## 一、登录信息

| 项目 | 值 |
|------|-----|
| **集群地址** | `login.alice.universiteitleiden.nl` |
| **SSH 命令** | `ssh <ULCN用户名>@login.alice.universiteitleiden.nl` |
| **VPN** | EduVPN → Leiden University → ULCN 登录 |
| **项目目录** | `~/CellSam` |
| **Conda 环境** | `cellsam` |

---

## 二、GPU 分区

| 分区 | GPU | 显存 | 时间限制 | 推荐场景 |
|------|-----|------|----------|----------|
| `gpu-short` | 混合 | - | 4h | 测试调试 |
| `gpu-l4-24g` | L4 | 24GB | 7天 | 常规训练 |
| **`gpu-a100-80g`** | **A100** | **80GB** | **7天** | **⭐ 推荐** |

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
```

---

## 四、CellSAM 训练环境配置 (一次性)

```bash
module load Miniforge3/24.3.0-0
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

conda activate cellsam
export PYTHONPATH=$PYTHONPATH:~/CellSam/cellSAM_source
cd ~/CellSam
mkdir -p logs checkpoints

python src/train.py --config src/config/CONFIG_FILE.yaml
```
