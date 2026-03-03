#!/usr/bin/env python3
"""T17: Plot training curves from parsed CSV files.

Produces a 2×2 subplot figure (per R1 review):
  (a) Train Loss vs Epoch
  (b) Val PQ vs Epoch
  (c) Val BM-Dice vs Epoch
  (d) Val Semantic Dice vs Epoch

Each plot marks best epoch and early stop point.
X-axis shows both epoch and estimated wall-clock hours (R1 requirement).

Usage:
    # Single experiment
    python tools/plot_training_curves.py figures/csv/phase1_l4.csv -o figures/training_curves_phase1.png

    # Compare two experiments (Phase1 vs Best Config)
    python tools/plot_training_curves.py figures/csv/phase1_l4.csv figures/csv/best_config_l4.csv \\
        --labels "Phase 1 (L4)" "Best Config (L4)" -o figures/training_curves_comparison.png
"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Style configuration
COLORS = [
    "#2563eb",   # blue
    "#dc2626",   # red
    "#16a34a",   # green
    "#9333ea",   # purple
    "#ea580c",   # orange
]
STYLE_KWARGS = dict(linewidth=1.8, markersize=0)


def load_csv(path: Path) -> dict:
    """Load CSV into dict of lists."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                if k == "is_best_pq":
                    data[k].append(v.lower() == "true")
                elif v == "" or v is None:
                    data[k].append(None)
                else:
                    try:
                        data[k].append(float(v))
                    except ValueError:
                        data[k].append(v)
    return data


def find_best_epoch(data: dict) -> int | None:
    """Find the last epoch marked as best PQ."""
    best_epochs = [int(data["epoch"][i]) for i, b in enumerate(data["is_best_pq"]) if b]
    return best_epochs[-1] if best_epochs else None


def plot_curves(datasets: list[tuple[str, dict]], output_path: Path,
                title: str = "Training Curves", patience: int = 15):
    """Plot 2×2 training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    
    # Subplot definitions: (ax_position, y_key, ylabel, higher_better)
    subplots = [
        ((0, 0), "train_loss", "Train Loss", False),
        ((0, 1), "val_pq", "Val PQ@0.5", True),
        ((1, 0), "val_bm_dice", "Val BM-1to1 Dice", True),
        ((1, 1), "val_sem_dice", "Val Semantic Dice", True),
    ]
    
    for (r, c), y_key, ylabel, higher_better in subplots:
        ax = axes[r][c]
        
        for idx, (label, data) in enumerate(datasets):
            color = COLORS[idx % len(COLORS)]
            epochs = [int(e) for e in data["epoch"]]
            values = data[y_key]
            
            # Main curve
            ax.plot(epochs, values, color=color, label=label,
                    alpha=0.85, **STYLE_KWARGS)
            
            # Mark best epoch
            best_ep = find_best_epoch(data)
            if best_ep is not None and y_key != "train_loss":
                best_idx = epochs.index(best_ep)
                best_val = values[best_idx]
                ax.axvline(best_ep, color=color, linestyle=":", alpha=0.4, linewidth=1)
                ax.scatter([best_ep], [best_val], color=color, s=60, zorder=5,
                          marker="*", edgecolors="white", linewidth=0.5)
                ax.annotate(f"best={best_val:.3f}\nep{best_ep}",
                           xy=(best_ep, best_val), fontsize=7,
                           xytext=(8, -12 if idx == 0 else 12),
                           textcoords="offset points", color=color,
                           fontweight="bold", alpha=0.8)
            
            # Mark early stop point (best_ep + patience)
            if best_ep is not None and y_key == "val_pq":
                es_ep = best_ep + patience
                if es_ep <= max(epochs):
                    ax.axvline(es_ep, color=color, linestyle="--", alpha=0.3, linewidth=1)
                    ax.text(es_ep + 0.5, ax.get_ylim()[0] + 0.02,
                           f"ES@{es_ep}", fontsize=6, color=color, alpha=0.6,
                           rotation=90, va="bottom")
        
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.legend(fontsize=8, loc="best", framealpha=0.8)
        
        # Add wall-clock secondary x-axis if data available
        if datasets[0][1]["wall_hours"][0] is not None:
            max_hours = max(h for h in datasets[0][1]["wall_hours"] if h is not None)
            max_epoch = max(int(e) for e in datasets[0][1]["epoch"])
            
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] * max_hours / max_epoch if max_epoch > 0 else 0,
                        ax.get_xlim()[1] * max_hours / max_epoch if max_epoch > 0 else 0)
            ax2.set_xlabel("Wall-clock (hours)", fontsize=8, alpha=0.6)
            ax2.tick_params(labelsize=7, colors="gray")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"  → Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot T17 training curves")
    parser.add_argument("csvs", nargs="+", type=Path, help="CSV file(s) from parse_training_log.py")
    parser.add_argument("--labels", nargs="+", help="Labels for each CSV (default: filename stem)")
    parser.add_argument("-o", "--output", type=Path, default=Path("figures/training_curves.png"),
                        help="Output PNG path")
    parser.add_argument("--title", type=str, default="CellSAM Training Curves",
                        help="Figure title")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (for ES marker)")
    args = parser.parse_args()
    
    # Load datasets
    datasets = []
    labels = args.labels or [p.stem for p in args.csvs]
    if len(labels) < len(args.csvs):
        labels.extend([p.stem for p in args.csvs[len(labels):]])
    
    for csv_path, label in zip(args.csvs, labels):
        if not csv_path.exists():
            print(f"  ✗ {csv_path} not found", file=__import__("sys").stderr)
            continue
        data = load_csv(csv_path)
        datasets.append((label, data))
        print(f"  ✓ Loaded {label}: {len(data['epoch'])} epochs")
    
    if not datasets:
        print("No data loaded. Exiting.")
        return
    
    plot_curves(datasets, args.output, title=args.title, patience=args.patience)


if __name__ == "__main__":
    main()
