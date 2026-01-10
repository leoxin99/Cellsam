# ALICE HPC 连接与训练流程指南

## 一、连接 EduVPN

1. 打开 **EduVPN** 客户端
2. 选择 **Leiden University**
3. 使用 ULCN 账户登录
4. **保持 VPN 连接**

---

## 二、SSH 登录 ALICE

```powershell
# Windows PowerShell
ssh username@login.alice.universiteitleiden.nl

# 或使用跳板机
ssh -J ssh-gw.alice.universiteitleiden.nl username@login.alice.universiteitleiden.nl
```

将 `username` 替换为您的 ULCN 用户名。

---

## 三、查看可用分区

```bash
sinfo                    # 所有分区
sinfo -p gpu-l4-24g      # 特定分区状态
squeue -p gpu-l4-24g     # 队列任务
```

---

## 四、GPU 分区选择

| 分区 | GPU | 显存 | 时间 | 推荐度 |
|-----|-----|------|------|-------|
| `gpu-l4-24g` | L4 | 24GB | 7天 | ⭐⭐⭐ |
| `gpu-short` | 多种 | - | 4时 | ⭐⭐ (测试) |
| `gpu-2080ti-11g` | 2080Ti | 11GB | 7天 | ⭐ |
| `gpu-a100-80g` | A100 | 80GB | 7天 | ⭐⭐⭐ |

**CellSAM 推荐**: `gpu-l4-24g` (24GB 足够)

---

## 五、测试申请

```bash
# 交互式 GPU 会话 (测试用)
srun --partition=gpu-short --gres=gpu:l4:1 --time=00:30:00 --pty bash
```

---

## 六、训练脚本 (train_cellsam.sh)

```bash
#!/bin/bash
#SBATCH --job-name=cellsam
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=cellsam_%j.log

module load anaconda3/2023.07
source activate cellsam

cd ~/CellSam
python train_expanded.py --epochs 50 --batch_size 4
```

提交: `sbatch train_cellsam.sh`

---

## 七、监控任务

```bash
squeue -u $USER              # 查看任务
sacct -j <JOB_ID>            # 任务详情
tail -f cellsam_<JOB_ID>.log # 实时日志
scancel <JOB_ID>             # 取消任务
```

---

## 八、快速命令参考

| 命令 | 用途 |
|-----|------|
| `sinfo` | 分区状态 |
| `squeue -u $USER` | 我的任务 |
| `sbatch script.sh` | 提交任务 |
| `scancel <ID>` | 取消任务 |
| `srun --pty bash` | 交互会话 |
