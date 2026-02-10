"""
统一指标模块

提供训练验证和推理评估共用的指标计算函数。

命名规范:
  - BM-1to1:     compute_bm_1to1_dice     (主指标)
  - BM-Coverage: compute_bm_coverage_dice  (辅助诊断)
  - PQ:          compute_pq
  - AJI:         compute_aji
"""
from .instance_metrics import (
    compute_bm_1to1_dice,
    compute_bm_coverage_dice,
    compute_pq,
    compute_aji,
    compute_semantic_dice,
    compute_all_metrics,
)

__all__ = [
    'compute_bm_1to1_dice',
    'compute_bm_coverage_dice',
    'compute_pq',
    'compute_aji',
    'compute_semantic_dice',
    'compute_all_metrics',
]
