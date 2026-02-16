"""Full validation: count how many GT regions are now correctly matched to boxes across ALL splits."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'cellSAM_source')
import numpy as np
from skimage import measure
from augmented_dataset import AugmentedAllenDataset, load_split_ids

total_missed = 0
total_regions = 0
missed_details = []

for split in ['train', 'val', 'test']:
    ids = load_split_ids(split, 'data/splits')
    ds = AugmentedAllenDataset(data_dir='data/processed', is_training=False, sample_ids=ids, use_bf_only=False)
    
    split_missed = 0
    split_regions = 0
    
    for idx in range(len(ds)):
        s = ds[idx]
        mask = s['mask'].numpy().astype(np.int32)
        nb = int((s['boxes'][:s['num_boxes']].sum(dim=1) > 0).sum())
        nr = len(measure.regionprops(mask))
        
        split_regions += nr
        if nb != nr:
            diff = nr - nb
            split_missed += diff
            missed_details.append((split, s['sample_id'], nr, nb, diff))
    
    total_missed += split_missed
    total_regions += split_regions
    print(f"[{split:5s}] samples={len(ds):3d}  regions={split_regions:5d}  boxes_generated={split_regions - split_missed:5d}  missed={split_missed}")

print(f"\nTotal: regions={total_regions}, missed={total_missed}")
if missed_details:
    print("\nMissed samples:")
    for split, sid, nr, nb, diff in missed_details:
        print(f"  [{split}] {sid}: {nr} regions but {nb} boxes (missing {diff})")
else:
    print("\n✅ All GT regions matched to boxes — ZERO missing!")
