"""Compute F1_micro/P/R for all existing experiment result JSONs that have TP/FP/FN.

Note: RQ is per-image macro average from compute_all_metrics().
      F1_micro is global micro-F1 from aggregated TP/FP/FN.
      These are related but not identical.
"""
import json
import glob
import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

results = []
for f in sorted(glob.glob('experiments/*/results*.json')):
    data = json.load(open(f))
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict) and 'tp_total' in val:
                tp = val['tp_total']
                fp = val['fp_total']
                fn = val['fn_total']
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                results.append({
                    'file': f.replace('\\', '/'),
                    'key': key,
                    'arm': val.get('arm', ''),
                    'n': val.get('n_samples', 0),
                    'PQ': round(val.get('pq_mean', 0), 4),
                    'SQ': round(val.get('sq_mean', 0), 4),
                    'RQ_macro': round(val.get('rq_mean', 0), 4),
                    'F1_micro': round(f1, 4),
                    'P': round(p, 4),
                    'R': round(r, 4),
                    'BM_Dice': round(val.get('bm_1to1_dice_mean', 0), 4),
                    'AJI': round(val.get('aji_mean', 0), 4),
                    'TP': tp, 'FP': fp, 'FN': fn
                })

# Save JSON
with open('experiments/all_metrics_summary.json', 'w') as out:
    json.dump(results, out, indent=2)

# Print table
print(f"Collected {len(results)} result sets\n")
print(f"{'Experiment':<45} {'Key':<12} {'PQ':>6} {'SQ':>6} {'RQ_m':>6} {'F1_u':>6} {'P':>6} {'R':>6} {'BM':>6} {'AJI':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'N':>3}")
print("-" * 140)
for r in results:
    exp = r['file'].split('/')[1] if '/' in r['file'] else r['file']
    print(f"{exp:<45} {r['key']:<12} {r['PQ']:>6.4f} {r['SQ']:>6.4f} {r['RQ_macro']:>6.4f} {r['F1_micro']:>6.4f} {r['P']:>6.4f} {r['R']:>6.4f} {r['BM_Dice']:>6.4f} {r['AJI']:>6.4f} {r['TP']:>4} {r['FP']:>4} {r['FN']:>4} {r['n']:>3}")
