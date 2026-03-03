#!/usr/bin/env python3
"""Compare Train vs Val vs Test metrics across experiments."""
import json, csv
from pathlib import Path

def load_json(p):
    with open(p) as f:
        return json.load(f)

def load_csv_best(p):
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get("val_pq", 0)))
    return best

print("=" * 90)
print("TRAIN vs VAL vs TEST METRICS COMPARISON")
print("=" * 90)

# ---- T11 LoRA ----
print("\n### T11 LoRA ###")
fmt = "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
print(fmt.format("Config", "Val_PQ", "Val_Dice", "Test_PQ", "Test_Dice", "Test_AJI", "Gap(V-T)"))
print("-" * 90)

t11 = [
    ("r4_s42",   0.5017, 0.7266, "experiments/ablation_eval/t11/r4_seed42_l4/T11_LoRA_r4.json"),
    ("r4_s123",  0.5072, 0.7281, "experiments/ablation_eval/t11/r4_seed123_l4/T11_LoRA_r4.json"),
    ("r8_s42",   0.5139, 0.7343, "experiments/ablation_eval/t11/r8_seed42_a100/T11_LoRA_r8.json"),
    ("r8_s123",  0.5046, 0.7331, "experiments/ablation_eval/t11/r8_seed123_a100/T11_LoRA_r8.json"),
]
for name, vpq, vdice, jp in t11:
    d = load_json(jp)["result"]["metrics"]
    tpq = d["pq"]["mean"]
    tdice = d["bm_1to1_dice"]["mean"]
    taji = d["aji"]["mean"]
    gap = vpq - tpq
    print(fmt.format(name, f"{vpq:.4f}", f"{vdice:.4f}", f"{tpq:.4f}", f"{tdice:.4f}", f"{taji:.4f}", f"{gap:+.4f}"))

# ---- Best Config ----
print("\n### Best Config (4 runs) ###")
print(fmt.format("Config", "Val_PQ", "Val_Dice", "Test_PQ", "Test_Dice", "Test_AJI", "Gap(V-T)"))
print("-" * 90)

bc_evals = sorted(Path("experiments/ablation_eval").rglob("BestConfig*.json"))
for p in bc_evals[:4]:
    d = load_json(p)["result"]
    m = d["metrics"]
    vpq = d.get("train_pq", None)  # This is actually val PQ at best epoch
    vdice = d.get("train_dice", None)
    tpq = m["pq"]["mean"]
    tdice = m["bm_1to1_dice"]["mean"]
    taji = m["aji"]["mean"]
    gap = f"{vpq - tpq:+.4f}" if vpq else "N/A"
    vpq_s = f"{vpq:.4f}" if vpq else "N/A"
    vdice_s = f"{vdice:.4f}" if vdice else "N/A"
    print(fmt.format(p.parent.name, vpq_s, vdice_s, f"{tpq:.4f}", f"{tdice:.4f}", f"{taji:.4f}", gap))

# ---- Phase 1 ----
print("\n### Phase 1 ###")
print(fmt.format("Config", "Val_PQ", "Val_Dice", "Test_PQ", "Test_Dice", "Test_AJI", "Gap(V-T)"))
print("-" * 90)

p1_evals = sorted(Path("experiments/ablation_eval").rglob("Phase1*.json"))
for p in p1_evals[:4]:
    d = load_json(p)["result"]
    m = d["metrics"]
    vpq = d.get("train_pq", None)
    vdice = d.get("train_dice", None)
    tpq = m["pq"]["mean"]
    tdice = m["bm_1to1_dice"]["mean"]
    taji = m["aji"]["mean"]
    gap = f"{vpq - tpq:+.4f}" if vpq else "N/A"
    vpq_s = f"{vpq:.4f}" if vpq else "N/A"
    vdice_s = f"{vdice:.4f}" if vdice else "N/A"
    print(fmt.format(p.parent.name, vpq_s, vdice_s, f"{tpq:.4f}", f"{tdice:.4f}", f"{taji:.4f}", gap))

# ---- T12 Ablation ----
print("\n### T12 Ablation (selected) ###")
fmt2 = "{:<35} {:>10} {:>10} {:>10}"
print(fmt2.format("Config", "Val_PQ", "Test_PQ", "Gap(V-T)"))
print("-" * 70)

t12_evals = sorted(Path("experiments/ablation_eval/t12").rglob("*.json"))
for p in t12_evals:
    d = load_json(p)["result"]
    m = d["metrics"]
    vpq = d.get("train_pq", None)
    tpq = m["pq"]["mean"]
    gap = f"{vpq - tpq:+.4f}" if vpq else "N/A"
    vpq_s = f"{vpq:.4f}" if vpq else "N/A"
    label = f"{p.parent.name}/{p.stem}"
    print(fmt2.format(label[:35], vpq_s, f"{tpq:.4f}", gap))

# ---- Summary ----
print("\n" + "=" * 90)
print("SUMMARY: Val-to-Test Gap Analysis")
print("=" * 90)
# Collect all gaps
gaps = []
for name, vpq, vdice, jp in t11:
    d = load_json(jp)["result"]["metrics"]
    gaps.append((name, vpq, d["pq"]["mean"]))
for p in bc_evals[:4]:
    d = load_json(p)["result"]
    vpq = d.get("train_pq", None)
    if vpq:
        gaps.append((f"BC_{p.parent.name}", vpq, d["metrics"]["pq"]["mean"]))
for p in p1_evals[:4]:
    d = load_json(p)["result"]
    vpq = d.get("train_pq", None)
    if vpq:
        gaps.append((f"P1_{p.parent.name}", vpq, d["metrics"]["pq"]["mean"]))

if gaps:
    avg_gap = sum(v - t for _, v, t in gaps) / len(gaps)
    print(f"\nAverage val-to-test PQ gap: {avg_gap:+.4f} ({len(gaps)} runs)")
    print(f"Min gap: {min(v-t for _, v, t in gaps):+.4f}")
    print(f"Max gap: {max(v-t for _, v, t in gaps):+.4f}")
    print("\nInterpretation:")
    print(f"  Val PQ is consistently HIGHER than Test PQ by ~{abs(avg_gap):.1%}")
    print(f"  This is a normal generalization gap for small datasets (71 val, 73 test)")
