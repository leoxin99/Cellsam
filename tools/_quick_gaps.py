import json
from pathlib import Path

# T12 ablation
for p in sorted(Path("experiments/ablation_eval").glob("*.json")):
    d = json.load(open(p))["result"]
    m = d["metrics"]
    vpq = d.get("train_pq", "N/A")
    tpq = m["pq"]["mean"]
    gap = f"{vpq-tpq:+.4f}" if isinstance(vpq, float) else "N/A"
    vstr = f"{vpq:.4f}" if isinstance(vpq, float) else vpq
    print(f"{p.stem:<30} val_PQ={vstr:<8} test_PQ={tpq:.4f} gap={gap}")

# Best Config 4 runs
print("\n--- Best Config 4 runs ---")
for p in sorted(Path("experiments/ablation_eval").rglob("BestConfig*.json")):
    d = json.load(open(p))["result"]
    m = d["metrics"]
    vpq = d.get("train_pq", "N/A")
    tpq = m["pq"]["mean"]
    gap = f"{vpq-tpq:+.4f}" if isinstance(vpq, float) else "N/A"
    vstr = f"{vpq:.4f}" if isinstance(vpq, float) else vpq
    print(f"  {p.parent.name:<20} val_PQ={vstr:<8} test_PQ={tpq:.4f} gap={gap}")

# T11
print("\n--- T11 LoRA ---")
for p in sorted(Path("experiments/ablation_eval/t11").rglob("*.json")):
    d = json.load(open(p))["result"]
    m = d["metrics"]
    vpq = d.get("train_pq", "N/A")
    tpq = m["pq"]["mean"]
    gap = f"{vpq-tpq:+.4f}" if isinstance(vpq, float) else "N/A"
    vstr = f"{vpq:.4f}" if isinstance(vpq, float) else vpq
    print(f"  {p.parent.name:<20} val_PQ={vstr:<8} test_PQ={tpq:.4f} gap={gap}")
