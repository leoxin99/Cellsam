"""Quick: compute global F1 for MedSAM and CellSAM baseline."""
import json, os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# MedSAM
d = json.load(open('experiments/baseline_comparison/results.json'))
agg = d['methods']['medsam']['aggregated']
tp = agg['tp']['mean'] * 73
fp = agg['fp']['mean'] * 73
fn = agg['fn']['mean'] * 73
p = tp / (tp + fp) if tp + fp > 0 else 0
r = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * p * r / (p + r) if p + r > 0 else 0
print(f"MedSAM test73: PQ={agg['pq']['mean']:.4f}, SQ={agg['sq']['mean']:.4f}, RQ={agg['rq']['mean']:.4f}")
print(f"  BM-Dice={agg['bm_1to1_dice']['mean']:.4f}, AJI={agg['aji']['mean']:.4f}, SemDice={agg['semantic_dice']['mean']:.4f}")
print(f"  F1={f1:.4f}, P={p:.4f}, R={r:.4f}, TP={tp:.0f}, FP={fp:.0f}, FN={fn:.0f}")

# CellSAM official (combined results)
dc = json.load(open('experiments/baseline_comparison/results_combined.json'))
for name, details in dc.items():
    if isinstance(details, dict) and 'aggregated' in details:
        a = details['aggregated']
        if 'tp' in a:
            tp2 = a['tp']['mean'] * a.get('n_gt_cells', {}).get('n', 73)
            fp2 = a['fp']['mean'] * 73
            fn2 = a['fn']['mean'] * 73
            p2 = tp2 / (tp2 + fp2) if tp2 + fp2 > 0 else 0
            r2 = tp2 / (tp2 + fn2) if tp2 + fn2 > 0 else 0
            f2 = 2 * p2 * r2 / (p2 + r2) if p2 + r2 > 0 else 0
            print(f"\n{name}: PQ={a['pq']['mean']:.4f}, SQ={a['sq']['mean']:.4f}, RQ={a['rq']['mean']:.4f}")
            print(f"  BM-Dice={a['bm_1to1_dice']['mean']:.4f}, AJI={a['aji']['mean']:.4f}")
            print(f"  F1={f2:.4f}, P={p2:.4f}, R={r2:.4f}")

# T34 val results
t34 = json.load(open('experiments/t34_official_path_ablation/results_val.json'))
for arm, data in t34.items():
    if isinstance(data, dict) and 'pq_mean' in data:
        tp3 = data['tp_total']
        fp3 = data['fp_total']
        fn3 = data['fn_total']
        p3 = tp3 / (tp3 + fp3) if tp3 + fp3 > 0 else 0
        r3 = tp3 / (tp3 + fn3) if tp3 + fn3 > 0 else 0
        f3 = 2 * p3 * r3 / (p3 + r3) if p3 + r3 > 0 else 0
        print(f"\nT34 {arm} ({data.get('arm','')}): PQ={data['pq_mean']:.4f}, F1={f3:.4f}")
        print(f"  BM-Dice={data['bm_1to1_dice_mean']:.4f}, AJI={data['aji_mean']:.4f}")
