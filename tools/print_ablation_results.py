"""Compile T12 ablation results from seed42 + seed123 into final table."""
import json, glob, os

EXPS = [
    ("E_phase1_rebalance_l4", "Full (Phase1)"),
    ("Ab0_bce_dice_only", "Ab-0: BCE+Dice only"),
    ("Ab1_no_boundary", "Ab-1: w/o Boundary"),
    ("Ab2_no_contour", "Ab-2: w/o Contour"),
    ("Ab3_no_aji", "Ab-3: w/o AJI"),
    ("Ab4_no_pq_earlystop", "Ab-4: w/o PQ ES"),
    ("Ab5_posw10", "Ab-5: posw=10"),
]

METRICS = ["pq", "bm_1to1_dice", "aji", "semantic_dice"]
METRIC_NAMES = ["PQ", "BM-Dice", "AJI", "Sem.Dice"]

results = {}
for exp_id, label in EXPS:
    seeds = {}
    for seed_dir in ["seed42", "seed123"]:
        f = f"experiments/ablation_eval/{seed_dir}/{exp_id}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            m = d["result"]["metrics"]
            seeds[seed_dir] = {k: m[k]["mean"] for k in METRICS}
            seeds[seed_dir]["epoch"] = d["result"]["epoch"]
            seeds[seed_dir]["fp"] = m.get("fp", {}).get("mean", 0)
            seeds[seed_dir]["conflict"] = m.get("conflict_pixels", {}).get("mean", 0)
    results[exp_id] = {"label": label, "seeds": seeds}

# Print per-seed table
print("=" * 100)
print("T12 LOSS ABLATION — PER-SEED RESULTS")
print("=" * 100)
header = f"{'Experiment':30s} {'Seed':6s} {'Ep':>4s} {'PQ':>8s} {'BM-Dice':>8s} {'AJI':>8s} {'Sem.D':>8s} {'FP':>6s} {'Conflict':>8s}"
print(header)
print("-" * 100)
for exp_id, label in EXPS:
    for sd, sdata in results[exp_id]["seeds"].items():
        seed = sd.replace("seed", "")
        print(f"{label:30s} {seed:>6s} {sdata['epoch']:4d} {sdata['pq']:8.4f} {sdata['bm_1to1_dice']:8.4f} {sdata['aji']:8.4f} {sdata['semantic_dice']:8.4f} {sdata['fp']:6.1f} {sdata['conflict']:8.0f}")

# Print mean +/- diff table
print()
print("=" * 100)
print("T12 LOSS ABLATION — MEAN (2 seeds)")
print("=" * 100)
header2 = f"{'Experiment':30s} {'PQ':>12s} {'BM-Dice':>12s} {'AJI':>12s} {'Sem.Dice':>12s} {'dPQ':>8s}"
print(header2)
print("-" * 100)

baseline_pq = None
for exp_id, label in EXPS:
    seeds = results[exp_id]["seeds"]
    if len(seeds) == 2:
        vals = list(seeds.values())
        row = {}
        for k in METRICS:
            v1, v2 = vals[0][k], vals[1][k]
            mean = (v1 + v2) / 2
            diff = abs(v1 - v2) / 2
            row[k] = (mean, diff)
        
        pq_mean = row["pq"][0]
        if baseline_pq is None:
            baseline_pq = pq_mean
            dpq = "  ---"
        else:
            dpq = f"{(pq_mean - baseline_pq)*100:+.2f}pp"
        
        print(f"{label:30s} {row['pq'][0]:.4f}+/-{row['pq'][1]:.4f} {row['bm_1to1_dice'][0]:.4f}+/-{row['bm_1to1_dice'][1]:.4f} {row['aji'][0]:.4f}+/-{row['aji'][1]:.4f} {row['semantic_dice'][0]:.4f}+/-{row['semantic_dice'][1]:.4f} {dpq:>8s}")
    else:
        print(f"{label:30s}  (missing seed data)")

print()
print("NOTE: +/- shows half the range between 2 seeds (not std)")
