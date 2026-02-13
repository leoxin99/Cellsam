# -*- coding: utf-8 -*-
"""
训练前验证脚本（通用版，适配 Phase1/Phase2）。

用法:
    python tools/verify_training_config.py --config src/config/phase2a_neighbor_overlap.yaml
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def verify_files_exist(config_path: str, cfg: dict) -> bool:
    """验证必要文件存在。"""
    print("\n[文件检查]")
    splits_dir = _get_nested(cfg, "data.splits_dir", "data/splits")
    use_neighbor = bool(_get_nested(cfg, "loss.use_neighbor", False))
    use_overlap = bool(_get_nested(cfg, "loss.use_overlap", False))

    required_files = [
        "src/train.py",
        "src/losses/combined.py",
        "src/augmented_dataset.py",
        "src/inference/core.py",
        "tools/test_unified_regression.py",
        str(Path(splits_dir) / "train_ids.txt"),
        str(Path(splits_dir) / "val_ids.txt"),
        config_path,
    ]
    if use_neighbor or use_overlap:
        required_files.append("tools/test_loss_gradients.py")

    all_ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            _ok(file_path)
        else:
            _fail(f"{file_path} 不存在")
            all_ok = False
    return all_ok


def verify_dataset(
    cfg: dict,
    split: str = "val",
    n_samples: int = 3,
) -> bool:
    """验证数据加载、通道值、mask/boxes 结构。"""
    print("\n[数据加载检查]")
    from augmented_dataset import AugmentedAllenDataset

    splits_dir = Path(_get_nested(cfg, "data.splits_dir", "data/splits"))
    split_path = splits_dir / f"{split}_ids.txt"
    data_dir = _get_nested(cfg, "data.processed_data_dir", "data/processed")
    target_size = tuple(_get_nested(cfg, "data.target_size", [1024, 1024]))
    use_bf_only = bool(_get_nested(cfg, "data.use_bf_only", False))
    use_semantic_mapping = bool(_get_nested(cfg, "data.use_semantic_mapping", False))

    if not split_path.exists():
        _fail(f"split 文件不存在: {split_path}")
        return False

    sample_ids = split_path.read_text(encoding="utf-8").strip().splitlines()[:n_samples]
    if not sample_ids:
        _fail(f"split 文件为空: {split_path}")
        return False

    dataset = AugmentedAllenDataset(
        data_dir=data_dir,
        target_size=target_size,
        is_training=False,
        sample_ids=sample_ids,
        use_bf_only=use_bf_only,
        use_semantic_mapping=use_semantic_mapping,
    )
    _ok(f"Loaded {len(dataset)} samples (training=False)")
    all_ok = True

    for idx in range(min(n_samples, len(dataset))):
        sample = dataset[idx]
        print(f"  Sample {idx}: {sample['sample_id']}")

        image = sample["image"].numpy()
        mask = sample["mask"].numpy()
        num_boxes = int(sample["num_boxes"])

        if image.shape[0] != 3:
            _fail(f"image channels={image.shape[0]} (应为 3)")
            all_ok = False
        else:
            _ok(f"image shape={image.shape}")

        if tuple(image.shape[-2:]) != target_size:
            _fail(f"image spatial={tuple(image.shape[-2:])}, config target_size={target_size}")
            all_ok = False

        for c, name in enumerate(["BF", "DAPI", "Actn2"]):
            unique = int(len(np.unique(image[c])))
            if unique <= 10:
                _fail(f"{name} unique={unique} (应 > 10)")
                all_ok = False
            else:
                _ok(f"{name} unique={unique}")

        labels = int(len(np.unique(mask)) - 1)
        if labels < 1:
            _fail(f"mask labels={labels} (应 >= 1)")
            all_ok = False
        else:
            _ok(f"mask labels={labels}")

        if num_boxes < 1:
            _fail(f"num_boxes={num_boxes} (应 >= 1)")
            all_ok = False
        else:
            _ok(f"num_boxes={num_boxes}")

    return all_ok


def verify_inference_consistency() -> bool:
    """验证统一推理默认参数是否可用。"""
    print("\n[统一推理口径检查]")
    try:
        from inference.core import InferenceConfig
    except Exception as exc:  # pragma: no cover - guard
        _fail(f"无法导入 InferenceConfig: {exc}")
        return False

    cfg = InferenceConfig.default()
    ok = True

    if abs(float(cfg.box_expand) - 0.1) < 1e-8:
        _ok(f"InferenceConfig.default().box_expand={cfg.box_expand}")
    else:
        _warn(f"Inference box_expand={cfg.box_expand}（建议 0.1）")

    if cfg.conflict_policy in {"argmax_prob", "first_write", "last_write"}:
        _ok(f"conflict_policy={cfg.conflict_policy}")
    elif cfg.conflict_policy in {"first", "last"}:
        _warn(
            f"conflict_policy={cfg.conflict_policy} 为旧别名，"
            "建议改为 first_write / last_write"
        )
    else:
        _fail(f"未知 conflict_policy={cfg.conflict_policy}")
        ok = False

    return ok


def verify_train_runtime_assumptions(cfg: dict) -> bool:
    """验证 train.py 的关键运行前提（避免常见启动即失败）。"""
    print("\n[训练运行前提检查]")
    ok = True

    warmup_epochs = _get_nested(cfg, "training.warmup_epochs")
    if isinstance(warmup_epochs, int) and warmup_epochs >= 1:
        _ok(f"training.warmup_epochs={warmup_epochs}")
    else:
        _fail(f"training.warmup_epochs 非法: {warmup_epochs} (CosineAnnealingWarmRestarts 要求 >=1)")
        ok = False

    target_size = _get_nested(cfg, "data.target_size")
    if target_size == [1024, 1024]:
        _ok("data.target_size=[1024, 1024] (与 train.py 当前上采样实现一致)")
    else:
        _warn(
            "data.target_size 不是 [1024, 1024]。当前 train.py 内部仍硬编码上采样到 1024，"
            "如需改分辨率，先同步修改 train.py 对 low_res_masks 的插值尺寸。"
        )

    return ok


def _get_nested(config: dict, key: str, default=None):
    cur = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def verify_config_file(config_path: str) -> bool:
    """验证训练配置关键字段。"""
    print(f"\n[配置文件检查] {config_path}")
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        _fail("配置文件不存在")
        return False

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ok = True

    required_sections = ["data", "model", "training", "loss", "output"]
    for sec in required_sections:
        if sec in cfg:
            _ok(f"section: {sec}")
        else:
            _fail(f"缺少 section: {sec}")
            ok = False

    experiment_name = _get_nested(cfg, "output.experiment_name")
    if experiment_name:
        _ok(f"output.experiment_name={experiment_name}")
    else:
        _fail("output.experiment_name 缺失")
        ok = False

    box_expand = _get_nested(cfg, "loss.box_expand")
    if box_expand == 0.1:
        _ok("loss.box_expand=0.1")
    else:
        _warn(f"loss.box_expand={box_expand}（建议 0.1）")

    epochs = _get_nested(cfg, "training.epochs")
    batch_size = _get_nested(cfg, "training.batch_size")
    if isinstance(epochs, int) and epochs > 0:
        _ok(f"training.epochs={epochs}")
    else:
        _fail(f"training.epochs 非法: {epochs}")
        ok = False
    if isinstance(batch_size, int) and batch_size > 0:
        _ok(f"training.batch_size={batch_size}")
    else:
        _fail(f"training.batch_size 非法: {batch_size}")
        ok = False

    use_neighbor = bool(_get_nested(cfg, "loss.use_neighbor", False))
    use_overlap = bool(_get_nested(cfg, "loss.use_overlap", False))
    if use_neighbor:
        w = _get_nested(cfg, "loss.neighbor_weight")
        g = _get_nested(cfg, "loss.neighbor_gamma")
        if w is None or g is None:
            _fail("use_neighbor=true 但缺 neighbor_weight/neighbor_gamma")
            ok = False
        else:
            _ok(f"neighbor_weight={w}, neighbor_gamma={g}")
    if use_overlap:
        w = _get_nested(cfg, "loss.overlap_weight")
        m = _get_nested(cfg, "loss.overlap_margin")
        if w is None or m is None:
            _fail("use_overlap=true 但缺 overlap_weight/overlap_margin")
            ok = False
        else:
            _ok(f"overlap_weight={w}, overlap_margin={m}")

    return ok


def verify_slurm_scripts(slurm_dir: str = "scripts") -> bool:
    """检查 SLURM 脚本中的已知 Alice 环境陷阱。"""
    print(f"\n[SLURM 脚本检查] {slurm_dir}/")
    slurm_path = Path(slurm_dir)
    if not slurm_path.exists():
        _warn(f"{slurm_dir} 目录不存在，跳过 SLURM 检查")
        return True

    sh_files = list(slurm_path.glob("*.sh"))
    if not sh_files:
        _warn(f"{slurm_dir} 下无 .sh 文件")
        return True

    ok = True
    for sh_file in sh_files:
        content = sh_file.read_text(encoding="utf-8", errors="replace")
        if "#SBATCH" not in content:
            continue  # 非 SLURM 脚本

        print(f"  检查: {sh_file.name}")
        lines = content.splitlines()

        # 检查 1: 硬编码 ~/miniconda3 路径
        for i, line in enumerate(lines, 1):
            if "miniconda3" in line and not line.strip().startswith("#"):
                _fail(f"  L{i}: 硬编码 ~/miniconda3 路径（Alice 已迁移到 Miniforge3）")
                ok = False

        # 检查 2: 旧的 cuda module 名称（小写 cuda/）
        for i, line in enumerate(lines, 1):
            if re.search(r"module\s+load\s+cuda/", line, re.IGNORECASE):
                if "cuda/11" in line.lower():
                    _fail(f"  L{i}: 使用已移除的 cuda/11.x module（应为 CUDA/12.1.1）")
                    ok = False

        # 检查 3: set -u 在 conda activate 之前
        set_u_line = None
        conda_activate_line = None
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"set\s+-[a-z]*u", stripped) and set_u_line is None:
                set_u_line = i
            if "conda activate" in stripped and conda_activate_line is None:
                conda_activate_line = i

        if set_u_line and conda_activate_line and set_u_line < conda_activate_line:
            _fail(
                f"  set -u (L{set_u_line}) 在 conda activate (L{conda_activate_line}) 之前，"
                "conda MKL 脚本可能因 unbound variable 失败"
            )
            ok = False

        if ok:
            _ok(f"{sh_file.name}: 无已知陷阱")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/phase2a_neighbor_overlap.yaml",
        help="训练配置路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="数据检查使用的 split 名称",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="数据检查抽样数量",
    )
    parser.add_argument(
        "--slurm-dir",
        type=str,
        default="scripts",
        help="SLURM 脚本目录",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[FAIL] 配置文件不存在: {args.config}")
        return 1
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    print("=" * 60)
    print("CellSAM 训练前验证")
    print("=" * 60)

    results = [
        ("文件存在", verify_files_exist(args.config, cfg)),
        ("配置文件", verify_config_file(args.config)),
        ("数据加载", verify_dataset(cfg, split=args.split, n_samples=args.n_samples)),
        ("训练运行前提", verify_train_runtime_assumptions(cfg)),
        ("统一推理口径", verify_inference_consistency()),
        ("SLURM 脚本", verify_slurm_scripts(args.slurm_dir)),
    ]

    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    all_ok = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
        all_ok = all_ok and passed
    print("=" * 60)
    if all_ok:
        print("[OK] 所有检查通过，可以开始训练")
        return 0
    print("[FAIL] 存在检查失败项，请修复后再训练")
    return 1


if __name__ == "__main__":
    sys.exit(main())
