"""
CellSAM 权重对比工具 — model vs model_cp vs SAM ViT-B

背景 (2026-03-01, R1/Reviewer 创建):
  T24 审计发现我们的训练/推理始终使用 model.model.* 而非官方推理的 model_cp.*。
  为了搞清 CellSAM checkpoint (cellsam_general.pt) 中 model 和 model_cp 与
  Meta 原始 SAM ViT-B (sam_vit_b_01ec64.pth) 的关系, 创建此脚本做逐参数对比。

目的:
  1. 验证 CellSAM 的 model 是否 == 原始 SAM ViT-B (结论: 否, 0/314 相同)
  2. 验证 CellSAM 的 model_cp 是否 == 原始 SAM ViT-B (结论: 部分, 47/314 相同)
  3. 验证 model 与 model_cp 之间的差异 (结论: 314/314 全部不同)

依赖:
  - checkpoints/sam_vit_b_01ec64.pth (从 Meta 下载, ~375MB)
  - cellSAM_source/ (CellSAM 官方仓库)

结果记录在: docs/paper_preparation.md §2.1b
"""
import torch, sys
sys.path.insert(0, 'cellSAM_source')
from cellSAM import get_model
from segment_anything import sam_model_registry

print('Loading CellSAM...')
cellsam = get_model()
print(f'adv_mode = {cellsam.adv_mode}')

print('Loading original SAM ViT-B...')
sam = sam_model_registry['vit_b'](checkpoint='checkpoints/sam_vit_b_01ec64.pth')

# Compare model vs SAM
print('\n=== CellSAM .model vs original SAM ViT-B ===')
n_same, n_diff = 0, 0
diffs = []
for name in sam.state_dict():
    if name in cellsam.model.state_dict():
        d = (cellsam.model.state_dict()[name].float() - sam.state_dict()[name].float()).abs().max().item()
        if d < 1e-6:
            n_same += 1
        else:
            n_diff += 1
            diffs.append((name, d))

print(f'Identical: {n_same}')
print(f'Different: {n_diff}')
if diffs:
    print('Top 5 largest differences:')
    for name, d in sorted(diffs, key=lambda x: -x[1])[:5]:
        print(f'  {name}: max_diff={d:.6f}')
else:
    print('>> CONCLUSION: model == original SAM ViT-B (IDENTICAL)')

# Compare model_cp vs SAM
print('\n=== CellSAM .model_cp vs original SAM ViT-B ===')
n_same2, n_diff2 = 0, 0
for name in sam.state_dict():
    if name in cellsam.model_cp.state_dict():
        d = (cellsam.model_cp.state_dict()[name].float() - sam.state_dict()[name].float()).abs().max().item()
        if d < 1e-6:
            n_same2 += 1
        else:
            n_diff2 += 1

print(f'Identical: {n_same2}')
print(f'Different: {n_diff2}')
