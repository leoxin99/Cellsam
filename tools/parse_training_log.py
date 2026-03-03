#!/usr/bin/env python3
"""T17: Parse ALICE training log files into CSV for plotting.

Usage:
    python tools/parse_training_log.py logs/p1_l4_974531.log -o figures/csv/phase1_l4.csv
    python tools/parse_training_log.py logs/*.log -o figures/csv/  # batch mode (output dir)
"""
import re
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime


# Regex for epoch lines:
# Epoch [1/50] Train Loss: 0.5381, BM-1to1: 0.4248, BM-Cov: 0.4608, Gap: 0.0360, PQ: 0.0885, Sem: 0.6926, Conflict: 183126, LR: 0.000090
EPOCH_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] "
    r"Train Loss: ([\d.]+), "
    r"BM-1to1: ([\d.]+), "
    r"BM-Cov: ([\d.]+), "
    r"Gap: ([\d.]+), "
    r"PQ: ([\d.]+), "
    r"Sem: ([\d.]+), "
    r"Conflict: (\d+), "
    r"LR: ([\d.]+)"
)

# Best PQ marker
BEST_PQ_RE = re.compile(r"-> New best PQ!")

# Job metadata
JOB_DATE_RE = re.compile(r"Date: (.+)")
JOB_ID_RE = re.compile(r"Job ID: (\d+)")
GPU_RE = re.compile(r"GPU: (.+)")
END_RE = re.compile(r"End: (.+)")

CSV_HEADER = [
    "epoch", "total_epochs", "train_loss",
    "val_bm_dice", "val_bm_cov", "val_gap",
    "val_pq", "val_sem_dice", "val_conflict",
    "lr", "is_best_pq", "wall_hours"
]


def parse_log(log_path: Path) -> list[dict]:
    """Parse a single training log file into a list of epoch records."""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    
    records = []
    start_time = None
    current_best_epoch = None
    
    # Try to extract start time from header
    for line in lines[:15]:
        m = JOB_DATE_RE.search(line)
        if m:
            try:
                start_time = datetime.strptime(m.group(1).strip(), "%a %b %d %H:%M:%S %Z %Y")
            except ValueError:
                pass
            break
    
    # Try to extract end time for wall-clock estimation
    end_time = None
    for line in lines[-5:]:
        m = END_RE.search(line)
        if m:
            try:
                end_time = datetime.strptime(m.group(1).strip(), "%a %b %d %H:%M:%S %Z %Y")
            except ValueError:
                pass
            break
    
    # Parse epoch lines
    i = 0
    while i < len(lines):
        line = lines[i]
        m = EPOCH_RE.search(line)
        if m:
            epoch = int(m.group(1))
            total = int(m.group(2))
            
            # Check if next line is best PQ marker
            is_best = False
            if i + 1 < len(lines) and BEST_PQ_RE.search(lines[i + 1]):
                is_best = True
                current_best_epoch = epoch
            
            # Estimate wall-clock hours (linear interpolation)
            wall_hours = None
            if start_time and end_time and total > 0:
                total_hours = (end_time - start_time).total_seconds() / 3600
                wall_hours = round(total_hours * epoch / total, 2)
            
            records.append({
                "epoch": epoch,
                "total_epochs": total,
                "train_loss": float(m.group(3)),
                "val_bm_dice": float(m.group(4)),
                "val_bm_cov": float(m.group(5)),
                "val_gap": float(m.group(6)),
                "val_pq": float(m.group(7)),
                "val_sem_dice": float(m.group(8)),
                "val_conflict": int(m.group(9)),
                "lr": float(m.group(10)),
                "is_best_pq": is_best,
                "wall_hours": wall_hours,
            })
        i += 1
    
    return records


def write_csv(records: list[dict], output_path: Path):
    """Write parsed records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(records)
    print(f"  → {output_path} ({len(records)} epochs)")


def main():
    parser = argparse.ArgumentParser(description="Parse ALICE training logs to CSV")
    parser.add_argument("logs", nargs="+", type=Path, help="Log file(s) to parse")
    parser.add_argument("-o", "--output", type=Path, default=Path("figures/csv"),
                        help="Output CSV file or directory (default: figures/csv/)")
    args = parser.parse_args()
    
    for log_path in args.logs:
        if not log_path.exists():
            print(f"  ✗ {log_path} not found, skipping", file=sys.stderr)
            continue
        
        records = parse_log(log_path)
        if not records:
            print(f"  ✗ {log_path} — no epoch data found", file=sys.stderr)
            continue
        
        # Determine output path
        if args.output.suffix == ".csv":
            out = args.output
        else:
            out = args.output / f"{log_path.stem}.csv"
        
        write_csv(records, out)
    
    print(f"\nDone. Parsed {len(args.logs)} log(s).")


if __name__ == "__main__":
    main()
