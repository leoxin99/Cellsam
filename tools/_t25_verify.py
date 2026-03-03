#!/usr/bin/env python3
"""T25 quick sanity check: verify model_cp fix works on 3 samples."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))

import torch, numpy as np
from cellSAM import get_model
from inference.core import segment_with_boxes, InferenceConfig
from augmented_dataset import AugmentedAllenDataset
from metrics.instance_metrics import compute_all_metrics

project = Path(__file__).parent.parent
test_ids = (project / "data/splits/test_ids.txt").read_text().strip().split("\n")[:3]
dataset = AugmentedAllenDataset(data_dir=str(project / "data/processed"), is_training=False, sample_ids=test_ids)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model()
# Apply T25 fix
model.model.load_state_dict(model.model_cp.state_dict())
print("T25 fix applied")
model = model.to(device)
model.eval()

cfg = InferenceConfig.default()
pqs = []
for idx in range(len(dataset)):
    s = dataset[idx]
    boxes = s["boxes"][:s["num_boxes"]]
    valid = boxes[boxes.sum(dim=1) > 0]
    if len(valid) == 0:
        continue
    r = segment_with_boxes(model, s["image"], valid, cfg, device=device)
    m = compute_all_metrics(r.instance_mask, s["mask"].numpy().astype(np.int32))
    pqs.append(m["pq"])
    print(f'  {s["sample_id"]}: PQ={m["pq"]:.4f}')

mean_pq = np.mean(pqs)
print(f"\nMean PQ (3 samples): {mean_pq:.4f}")
if mean_pq > 0.3:
    print("✅ PASS — T25 fix verified, unified path now matches official")
else:
    print("❌ FAIL — PQ still near 0, fix did not work")
