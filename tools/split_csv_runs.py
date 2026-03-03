"""Split CSV files that contain multiple training runs into single-run CSVs."""
import csv
import shutil
import sys

def split_csv(input_path, output_path):
    with open(input_path) as f:
        reader = list(csv.DictReader(f))
        header = list(reader[0].keys())

    # Find split point: where epoch resets
    split = None
    for i in range(1, len(reader)):
        if int(reader[i]['epoch']) <= int(reader[i-1]['epoch']):
            split = i
            break

    if split is None:
        shutil.copy(input_path, output_path)
        print(f"  {input_path}: no split needed ({len(reader)} rows)")
        return

    run1 = reader[:split]
    run2 = reader[split:]
    print(f"  {input_path}: split at row {split}")
    print(f"    Run1: epochs 1-{run1[-1]['epoch']} ({len(run1)} rows)")
    print(f"    Run2: epochs 1-{run2[-1]['epoch']} ({len(run2)} rows)")

    # Keep the longer run
    longer = run2 if len(run2) >= len(run1) else run1
    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(longer)
    print(f"    Saved: {output_path} ({len(longer)} rows)")


split_csv('figures/csv/best_cfg_1030301.csv', 'figures/csv/best_cfg_a100.csv')
split_csv('figures/csv/best_l4_1030302.csv', 'figures/csv/best_cfg_l4.csv')
split_csv('figures/csv/p1_l4_974531.csv', 'figures/csv/p1_l4.csv')
