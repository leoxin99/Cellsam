import json, glob, os

files = sorted(glob.glob('experiments/ablation_eval/*.json'))
print(f"Found {len(files)} eval results\n")
print(f"{'Name':35s} {'PQ':>8s} {'BM-Dice':>10s} {'AJI':>8s} {'Sem.Dice':>10s} {'FP':>6s}")
print('-' * 85)

for f in files:
    d = json.load(open(f))
    r = d['result']
    m = r['metrics']
    name = r['experiment_name']
    print(f"{name:35s} {m['pq']['mean']:8.4f} {m['bm_1to1_dice']['mean']:10.4f} {m['aji']['mean']:8.4f} {m['semantic_dice']['mean']:10.4f} {m.get('fp',{}).get('mean',0):6.1f}")
