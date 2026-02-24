"""Consolidate all baseline results into a combined summary."""
import json, numpy as np
from datetime import datetime

METRIC_KEYS = ['bm_1to1_dice', 'bm_coverage_dice', 'gap_dice', 'pq', 'sq', 'rq', 'aji', 'semantic_dice', 'tp', 'fp', 'fn', 'n_gt_cells', 'n_pred_cells']

def agg(data):
    r = {}
    for k in METRIC_KEYS:
        vals = [d[k] for d in data if k in d and not isinstance(d.get(k), str)]
        if vals:
            r[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}
    return r

methods = {
    'cellpose_v4': json.load(open('experiments/baseline_comparison/per_sample_cellpose_v4_default.json')),
    'cellsam_pretrained': json.load(open('experiments/baseline_comparison/per_sample_cellsam_pretrained.json')),
    'sam_vit_b': json.load(open('experiments/baseline_comparison/per_sample_medsam.json')),
}

combined = {
    'timestamp': datetime.now().isoformat(),
    'task': 'T16 Baseline Comparison',
    'test_set': 'test(73)',
    'methods': {name: {'aggregated': agg(data), 'n_samples': len(data)} for name, data in methods.items()}
}

with open('experiments/baseline_comparison/results_combined.json', 'w') as f:
    json.dump(combined, f, indent=2)

header = f"{'Method':<25} {'PQ':>8} {'BM-Dice':>10} {'AJI':>8} {'Sem.Dice':>10} {'TP':>5} {'FP':>5} {'FN':>5}"
print(header)
print('-' * len(header))
for name, data in methods.items():
    a = agg(data)
    pq = a.get('pq', {}).get('mean', 0)
    dice = a.get('bm_1to1_dice', {}).get('mean', 0)
    aji = a.get('aji', {}).get('mean', 0)
    sd = a.get('semantic_dice', {}).get('mean', 0)
    tp = a.get('tp', {}).get('mean', 0)
    fp = a.get('fp', {}).get('mean', 0)
    fn = a.get('fn', {}).get('mean', 0)
    print(f"{name:<25} {pq:>8.4f} {dice:>10.4f} {aji:>8.4f} {sd:>10.4f} {tp:>5.1f} {fp:>5.1f} {fn:>5.1f}")
print(f"{'CellSAM Ours (Phase1)':<25} {'0.4640':>8} {'0.6950':>10} {'0.5190':>8} {'0.8560':>10} {'7.3':>5} {'2.7':>5} {'2.7':>5}")
print("\nResults saved to experiments/baseline_comparison/results_combined.json")
