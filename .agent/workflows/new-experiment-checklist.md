---
description: 新实验提交前的 Pre-Flight Checklist — 避免 PYTHONPATH/git/checkpoint/freeze 等历史问题
---

# 新实验 Pre-Flight Checklist

> 每次新实验必须逐项检查，不可跳过。

## Phase 1: 设计 (本地)

- [ ] 明确实验目标: 测试什么假设 / 改变什么变量
- [ ] 找到参照实验 YAML (最相似的已完成实验)
- [ ] 复制参照 YAML → 新 YAML, **只改目标变量**
- [ ] 确认数据配置: `use_bf_only` / `use_semantic_mapping` / `use_official_encoding` / `official_r_channel`
- [ ] 确认 freeze 策略: 哪些模块可训练, 预期 trainable count
- [ ] 确认 loss 配置: 所有 weight 参数是否传参到 `train_one_epoch()`
- [ ] 确认 checkpoint 目录命名包含 seed (`experiment_name` 字段)\r\n- [ ] **🚨 防碰撞**: 不同 seed 的**同一实验**必须生成不同的 checkpoint 目录:\r\n  - 代码自动生成: `{exp_name}_seed{seed}_{timestamp}`\r\n  - SLURM 脚本的 `EXP_PREFIX` 也必须包含 seed, 如 `T29a_Official_BF_seed123`\r\n  - **禁止**: 多个 seed 共用同一个 `EXP_PREFIX` → 会导致 `best_model.pt` 被覆盖\r\n  - SLURM `--job-name` 也要包含 seed 以便 `squeue` 区分, 如 `t29a_s123_a100`

## Phase 2: 本地验证

// turbo
- [ ] 本地 dry-run: `python src/train.py --config src/config/xxx.yaml` (跑 1-2 batch 验证)
- [ ] 检查 stdout: trainable 参数数量 (与预期匹配)
- [ ] 检查 stdout: channel mapping 打印信息 (R=?, G=?, B=?)
- [ ] 检查 stdout: enabled losses 列表 (含 IoU Head?)
- [ ] 无 `ModuleNotFoundError` / `ImportError`

## Phase 3: ALICE 部署

- [ ] `git add -A` (不是 `git add -f` 单个文件!)
- [ ] `git status`: 确认新文件/修改在 staged 列表
- [ ] `git commit -m "..."` + `git push`
- [ ] SSH 到 ALICE: `git pull`
- [ ] 确认新文件存在: `ls src/config/新配置.yaml`
- [ ] 确认 SLURM 脚本: partition/time/mem/seed 正确\r\n- [ ] **🚨 wall-time 必须 `--time=12:00:00`** (不要 48h, 影响 backfill 排队速度)
- [ ] SLURM 脚本中 `PYTHONPATH` 包含 `${HOME}/CellSam/src`
- [ ] `sbatch scripts/train_xxx.sh`
- [ ] 等 1-2 分钟: `squeue -u $USER` 检查状态 (R/PD, 不是 FAILED)
- [ ] 如果 R: `head -20 logs/xxx_$JOBID.log` 确认正常启动

## Phase 4: 训练后

- [ ] 检查 `.log` 文件: `Training complete! Best Val Dice/PQ`
- [ ] 检查 `.err` 文件: 无 ERROR / Traceback
- [ ] 检查 checkpoint 目录: `best_model.pt` 存在
- [ ] 下载 `best_model.pt` 到本地: `scp alice:~/CellSam/checkpoints/xxx/best_model.pt .`
- [ ] 用 **5 个固定样本** 做 napari 可视化 (BF+DAPI+Actn2)
- [ ] 更新 `docs/experiments_log.md`

## 参考: 7 次历史失败

| # | 事件 | 被哪一步防止 |
|:-:|------|:-----------:|
| 1 | 新文件未 `git add` | Phase 3 |
| 2 | PYTHONPATH 缺 `src/` | Phase 3 |
| 3 | checkpoint 目录覆盖 | Phase 1 |
| 4 | freeze 逻辑不完整 | Phase 2 |
| 5 | 本地 CUDA OOM | Phase 2 |
| 6 | eval `ls -td` 取错 ckpt | Phase 4 |
| 7 | CUDA module 版本 | Phase 3 |
