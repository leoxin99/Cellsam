"""Consolidate baseline per-sample results into a correctly labeled summary.

This script does not run any model. It only reads existing
`experiments/baseline_comparison/per_sample_*.json` files and aggregates them.

Key fixes vs. the historical version:
1. `per_sample_medsam.json` is labeled as `medsam`, not `sam_vit_b`.
2. Legacy unified-path CellSAM and official-path CellSAM are separated.
3. The output records the exact source file for each method.
4. Historical vanilla SAM ViT-B is not synthesized unless a dedicated
   `per_sample_sam_vit_b*.json` file actually exists.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path("experiments/baseline_comparison")
OUTPUT_PATH = OUTPUT_DIR / "results_combined.json"

METRIC_KEYS = [
    "bm_1to1_dice",
    "bm_coverage_dice",
    "gap_dice",
    "pq",
    "sq",
    "rq",
    "aji",
    "semantic_dice",
    "tp",
    "fp",
    "fn",
    "n_gt_cells",
    "n_pred_cells",
]

SOURCE_SPECS = [
    {
        "name": "cellpose_v4",
        "path": OUTPUT_DIR / "per_sample_cellpose_v4_default.json",
        "description": "Cellpose v4 default unified model on test(73).",
    },
    {
        "name": "cellpose_v4_d200",
        "path": OUTPUT_DIR / "per_sample_cellpose_d200.json",
        "description": "Cellpose v4 with manual diameter=200 on test(73).",
    },
    {
        "name": "cellsam_pretrained_legacy_unified",
        "path": OUTPUT_DIR / "per_sample_cellsam_pretrained.json",
        "description": (
            "CellSAM pretrained under the legacy unified inference path "
            "(historical result, not official CellSAM predict path)."
        ),
    },
    {
        "name": "cellsam_pretrained_official",
        "path": OUTPUT_DIR / "per_sample_cellsam_official.json",
        "description": "CellSAM pretrained under the official CellSAM predict path.",
    },
    {
        "name": "medsam",
        "path": OUTPUT_DIR / "per_sample_medsam.json",
        "description": "MedSAM oracle baseline on test(73).",
    },
    {
        "name": "samcell_livecell",
        "path": OUTPUT_DIR / "per_sample_samcell_livecell.json",
        "description": "SAMCell LIVECell checkpoint baseline on test(73).",
    },
]


def aggregate_metrics(data: list[dict]) -> dict:
    """Aggregate mean/std/n for standard metric keys."""
    result = {}
    for key in METRIC_KEYS:
        values = [item[key] for item in data if key in item and not isinstance(item.get(key), str)]
        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
    return result


def load_sources() -> dict:
    """Load all available per-sample sources with correct labels."""
    methods = {}
    for spec in SOURCE_SPECS:
        if not spec["path"].exists():
            continue
        with spec["path"].open("r", encoding="utf8") as handle:
            data = json.load(handle)
        methods[spec["name"]] = {
            "source_file": str(spec["path"]).replace("\\", "/"),
            "description": spec["description"],
            "per_sample": data,
        }
    return methods


def build_output(methods: dict) -> dict:
    """Build the combined JSON payload."""
    return {
        "timestamp": datetime.now().isoformat(),
        "task": "T16 Baseline Comparison",
        "test_set": "test(73)",
        "notes": [
            "This file is a convenience aggregate built from existing per-sample JSON files.",
            "It does not rerun inference.",
            "Historical vanilla SAM ViT-B is not included unless a dedicated per-sample file exists.",
            "Use source_file for provenance of each row.",
        ],
        "methods": {
            name: {
                "source_file": payload["source_file"],
                "description": payload["description"],
                "aggregated": aggregate_metrics(payload["per_sample"]),
                "n_samples": len(payload["per_sample"]),
            }
            for name, payload in methods.items()
        },
    }


def print_summary(methods: dict) -> None:
    """Print a compact summary table."""
    header = (
        f"{'Method':<32} {'PQ':>8} {'BM-Dice':>10} "
        f"{'AJI':>8} {'Sem.Dice':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, payload in methods.items():
        agg = aggregate_metrics(payload["per_sample"])
        pq = agg.get("pq", {}).get("mean", 0.0)
        dice = agg.get("bm_1to1_dice", {}).get("mean", 0.0)
        aji = agg.get("aji", {}).get("mean", 0.0)
        semantic = agg.get("semantic_dice", {}).get("mean", 0.0)
        print(f"{name:<32} {pq:>8.4f} {dice:>10.4f} {aji:>8.4f} {semantic:>10.4f}")


def main() -> None:
    methods = load_sources()
    output = build_output(methods)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    print_summary(methods)
    print()
    print(f"Results saved to {OUTPUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()
