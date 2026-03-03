#!/usr/bin/env python3
"""Quick test: postprocess on vs off."""
import sys
from pathlib import Path
project = Path(__file__).parent.parent
sys.path.insert(0, str(project / "src"))
sys.path.insert(0, str(project / "cellSAM_source"))

import numpy as np
import torch
from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset
from inference.core import segment_with_boxes, InferenceConfig
from metrics.instance_metrics import compute_all_metrics

model = get_model()
model.adv_mode = True
model = model.to("cuda").eval()

test_ids = (project / "data/splits/test_ids.txt").read_text().strip().split("\n")
ds = AugmentedAllenDataset(data_dir=str(project / "data/processed"), is_training=False, sample_ids=test_ids)

cfg_on = InferenceConfig(apply_postprocess=True)
cfg_off = InferenceConfig(apply_postprocess=False)

s = ds[0]
img = s["image"]
gt = s["mask"].numpy().astype(np.int32)
boxes = s["boxes"][:s["num_boxes"]]
valid = boxes[boxes.sum(dim=1) > 0]

r_off = segment_with_boxes(model, img, valid, cfg_off, "cuda")
m_off = compute_all_metrics(r_off.instance_mask, gt)

r_on = segment_with_boxes(model, img, valid, cfg_on, "cuda")
m_on = compute_all_metrics(r_on.instance_mask, gt)

print("Without postprocess: PQ=%.4f BM-Dice=%.4f cells=%d" % (m_off["pq"], m_off["bm_1to1_dice"], r_off.n_instances))
print("With postprocess:    PQ=%.4f BM-Dice=%.4f cells=%d" % (m_on["pq"], m_on["bm_1to1_dice"], r_on.n_instances))
