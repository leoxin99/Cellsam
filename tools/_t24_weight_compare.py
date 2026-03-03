"""T24: Compare CellSAM model vs model_cp weights."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
from cellSAM import get_model

print("Loading CellSAM pretrained model...")
m = get_model()
print(f"adv_mode: {m.adv_mode}")
print(f"type(model): {type(m.model)}")
print(f"type(model_cp): {type(m.model_cp)}")

# Compare all components
for comp_name, comp_a, comp_b in [
    ("image_encoder", m.model.image_encoder, m.model_cp.image_encoder),
    ("mask_decoder", m.model.mask_decoder, m.model_cp.mask_decoder),
    ("prompt_encoder", m.model.prompt_encoder, m.model_cp.prompt_encoder),
]:
    diffs = 0
    total = 0
    max_d = 0.0
    sd_a = comp_a.state_dict()
    sd_b = comp_b.state_dict()
    for key in sd_a:
        total += 1
        d = (sd_a[key] - sd_b[key]).abs().max().item()
        if d > 0:
            diffs += 1
            max_d = max(max_d, d)
            if diffs <= 2:
                print(f"  DIFF {comp_name}.{key}: max_diff={d:.6f}")
    status = "IDENTICAL" if diffs == 0 else f"DIFFERENT ({diffs}/{total} params, max_diff={max_d:.6f})"
    print(f"=== {comp_name}: {status} ===")
    print()

# Summary
print("=" * 60)
if m.adv_mode:
    print("adv_mode=True → predict() uses model_cp, NOT model")
    print("Our baseline_eval uses model.model → WRONG branch!")
else:
    print("adv_mode=False → predict() uses model → same as ours")
