# -*- coding: utf-8 -*-
"""
训练前验证脚本

使用方法:
    python tools/verify_training_config.py

检查内容:
1. 数据加载 - 通道值、mask、boxes
2. Loss 配置 - expand 参数
3. 推理一致性

所有检查通过后才能开始训练!
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def verify_dataset():
    """验证数据加载正确性"""
    print("【数据加载检查】")
    
    from augmented_dataset import AugmentedAllenDataset
    
    # 加载测试样本
    test_ids = Path("data/splits/test_ids.txt").read_text().strip().split('\n')[:3]
    dataset = AugmentedAllenDataset('data/processed', is_training=False, sample_ids=test_ids)
    
    all_pass = True
    
    for idx in range(min(3, len(dataset))):
        sample = dataset[idx]
        print(f"\n  Sample {idx}: {sample['sample_id']}")
        
        # A1: 通道唯一值
        for c, name in enumerate(['BF', 'DAPI', 'Actn2']):
            unique = len(np.unique(sample['image'][c].numpy()))
            if unique <= 10:
                print(f"    ❌ {name}: {unique} unique values (应 > 10)")
                all_pass = False
            else:
                print(f"    ✅ {name}: {unique} unique values")
        
        # A2: Mask 标签
        mask_labels = len(np.unique(sample['mask'].numpy())) - 1  # 排除背景
        if mask_labels < 1:
            print(f"    ❌ Mask: {mask_labels} labels (应 >= 1)")
            all_pass = False
        else:
            print(f"    ✅ Mask: {mask_labels} cell labels")
        
        # A3: Boxes
        if sample['num_boxes'] < 1:
            print(f"    ❌ Boxes: {sample['num_boxes']} (应 >= 1)")
            all_pass = False
        else:
            print(f"    ✅ Boxes: {sample['num_boxes']}")
    
    return all_pass


def verify_loss_config():
    """验证 Loss 配置"""
    print("\n【Loss 配置检查】")
    
    # 读取源文件检查 expand 值
    loss_file = Path("src/losses/combined.py")
    if not loss_file.exists():
        print("  ❌ combined.py 不存在!")
        return False
    
    content = loss_file.read_text()
    
    # 检查 expand = 0.1
    if 'expand = 0.1' in content:
        print("  ✅ CombinedLoss expand = 0.1")
    elif 'expand = 0.2' in content:
        print("  ⚠️ CombinedLoss expand = 0.2 (旧值，建议改为 0.1)")
    else:
        print("  ❌ 无法确认 expand 值!")
        return False
    
    return True


def verify_inference_consistency():
    """验证推理配置一致性"""
    print("\n【推理一致性检查】")
    
    eval_file = Path("tools/comprehensive_eval.py")
    if not eval_file.exists():
        print("  ⚠️ comprehensive_eval.py 不存在 (可选)")
        return True
    
    content = eval_file.read_text()
    
    if 'expand = 0.1' in content:
        print("  ✅ comprehensive_eval expand = 0.1")
    elif 'expand = 0.2' in content:
        print("  ⚠️ comprehensive_eval expand = 0.2 (与训练不一致!)")
    
    return True


def verify_files_exist():
    """验证必要文件存在"""
    print("\n【文件检查】")
    
    required_files = [
        "src/train.py",
        "src/losses/combined.py",
        "src/augmented_dataset.py",
        # "cellSAM_source/sam_vit_b_01ec64.pth",  # 自动下载，不需要本地存在
        "data/splits/train_ids.txt",
        "data/splits/val_ids.txt",
    ]
    
    all_exist = True
    for f in required_files:
        path = Path(f)
        if path.exists():
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} 不存在!")
            all_exist = False
    
    return all_exist


def verify_config_file(config_path: str):
    """验证训练配置文件"""
    print(f"\n【配置文件检查: {config_path}】")
    
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"  ❌ 配置文件不存在!")
        return False
    
    import yaml
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # 检查关键配置
    checks = [
        ('experiment_name', lambda x: x is not None),
        ('loss.box_expand', lambda x: x == 0.1),
        ('training.epochs', lambda x: x is not None and x > 0),
    ]
    
    all_pass = True
    
    if 'experiment_name' in config:
        print(f"  ✅ experiment_name: {config['experiment_name']}")
    
    if 'loss' in config:
        expand = config['loss'].get('box_expand', 'N/A')
        if expand == 0.1:
            print(f"  ✅ loss.box_expand: {expand}")
        else:
            print(f"  ⚠️ loss.box_expand: {expand} (建议 0.1)")
    
    return all_pass


def main():
    print("=" * 60)
    print("CellSAM 训练配置验证")
    print("=" * 60)
    
    results = []
    
    # 1. 数据加载
    results.append(("数据加载", verify_dataset()))
    
    # 2. Loss 配置
    results.append(("Loss 配置", verify_loss_config()))
    
    # 3. 推理一致性
    results.append(("推理一致性", verify_inference_consistency()))
    
    # 4. 文件检查
    results.append(("文件存在", verify_files_exist()))
    
    # 5. 配置文件 (可选)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args, _ = parser.parse_known_args()
    
    if args.config:
        results.append(("配置文件", verify_config_file(args.config)))
    
    # 汇总
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print("=" * 60)
    if all_pass:
        print("✅ 所有检查通过! 可以开始训练")
        return 0
    else:
        print("❌ 存在问题! 请修复后再训练")
        return 1


if __name__ == "__main__":
    sys.exit(main())
