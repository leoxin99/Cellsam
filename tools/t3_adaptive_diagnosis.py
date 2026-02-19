#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T3 Adaptive degeneration diagnosis on val(71).

This script focuses only on B2/B3 sensitivity diagnosis:
- B2: min_zlines sweep (radius fixed at 200)
- B3: zline_threshold sweep (radius fixed at 200, min_zlines fixed at 5)

Outputs:
1) experiments/ablation_adaptive_val/diagnosis_t3.json
2) patch diagnosis fields into experiments/ablation_adaptive_val/results.json
"""

import json
from datetime import datetime
from pathlib import Path

from ablation_adaptive_params import (
    evaluate_adaptive_params,
    load_test_samples,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _round(x, ndigits=4):
    return round(float(x), ndigits)


def _summarize(rows):
    f1_values = [r["f1"] for r in rows]
    ar_values = [r["adaptive_ratio"] for r in rows]
    mz_values = [r["mean_zlines"] for r in rows]
    fb_values = [r["fallback_count"] for r in rows]

    return {
        "f1_min": _round(min(f1_values)),
        "f1_max": _round(max(f1_values)),
        "f1_range": _round(max(f1_values) - min(f1_values), 6),
        "adaptive_ratio_mean": _round(sum(ar_values) / len(ar_values)),
        "adaptive_ratio_min": _round(min(ar_values)),
        "adaptive_ratio_max": _round(max(ar_values)),
        "mean_zlines_mean": _round(sum(mz_values) / len(mz_values)),
        "mean_zlines_min": _round(min(mz_values)),
        "mean_zlines_max": _round(max(mz_values)),
        "fallback_count_mean": _round(sum(fb_values) / len(fb_values), 2),
        "fallback_count_min": int(min(fb_values)),
        "fallback_count_max": int(max(fb_values)),
    }


def _infer_cause(b2_summary, b3_summary):
    # F1 almost flat in both B2/B3 => insensitive.
    flat_b2 = b2_summary["f1_range"] <= 1e-4
    flat_b3 = b3_summary["f1_range"] <= 2e-3

    if not (flat_b2 and flat_b3):
        return (
            "partial_sensitive",
            "B2/B3 not fully flat; further sweep or per-sample inspection is required.",
        )

    # If adaptive_ratio is low and fallback_count is high, insensitive likely caused by fallback dominance.
    ar_mean = (b2_summary["adaptive_ratio_mean"] + b3_summary["adaptive_ratio_mean"]) / 2.0
    fb_mean = (b2_summary["fallback_count_mean"] + b3_summary["fallback_count_mean"]) / 2.0

    if ar_mean < 0.4:
        return (
            "fallback_dominant",
            "B2/B3 are flat and adaptive_ratio is low; insensitive behavior is mainly due to fallback dominance.",
        )

    return (
        "zline_saturated",
        "B2/B3 are flat while adaptive mode remains common; likely Z-line signal is saturated at current radius.",
    )


def main():
    data_dir = PROJECT_DIR / "data" / "processed"
    split_file = PROJECT_DIR / "data" / "splits" / "val_ids.txt"
    output_dir = PROJECT_DIR / "experiments" / "ablation_adaptive_val"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    diagnosis_path = output_dir / "diagnosis_t3.json"

    # Keep detector parameters aligned with E34 lock.
    min_nucleus_area = 1500
    max_nucleus_area = 20000
    search_radius = 200

    print("=" * 70)
    print("T3 Adaptive degeneration diagnosis (val71)")
    print("=" * 70)
    print("Loading val samples...")
    samples = load_test_samples(data_dir, split_file, n_samples=71)
    print("Loaded {} samples".format(len(samples)))

    b2_values = [5, 10, 15, 20, 30]
    b3_values = [0.01, 0.02, 0.03, 0.05, 0.1]

    b2_rows = []
    for v in b2_values:
        m = evaluate_adaptive_params(
            samples,
            search_radius=search_radius,
            min_zlines=v,
            zline_threshold=0.03,
            min_nucleus_area=min_nucleus_area,
            max_nucleus_area=max_nucleus_area,
        )
        m["value"] = v
        b2_rows.append(m)
        print(
            "B2 min_zlines={}: f1={:.4f}, ar={:.4f}, fb={}, mz={:.4f}".format(
                v, m["f1"], m["adaptive_ratio"], m["fallback_count"], m["mean_zlines"]
            )
        )

    b3_rows = []
    for v in b3_values:
        m = evaluate_adaptive_params(
            samples,
            search_radius=search_radius,
            min_zlines=5,
            zline_threshold=v,
            min_nucleus_area=min_nucleus_area,
            max_nucleus_area=max_nucleus_area,
        )
        m["value"] = v
        b3_rows.append(m)
        print(
            "B3 zline_threshold={}: f1={:.4f}, ar={:.4f}, fb={}, mz={:.4f}".format(
                v, m["f1"], m["adaptive_ratio"], m["fallback_count"], m["mean_zlines"]
            )
        )

    b2_summary = _summarize(b2_rows)
    b3_summary = _summarize(b3_rows)
    cause_code, cause_text = _infer_cause(b2_summary, b3_summary)

    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "split": "val",
        "n_samples": len(samples),
        "detector_params": {
            "min_nucleus_area": min_nucleus_area,
            "max_nucleus_area": max_nucleus_area,
            "search_radius": search_radius,
        },
        "b2_min_zlines": {
            "fixed": {"search_radius": search_radius, "zline_threshold": 0.03},
            "rows": b2_rows,
            "summary": b2_summary,
        },
        "b3_zline_threshold": {
            "fixed": {"search_radius": search_radius, "min_zlines": 5},
            "rows": b3_rows,
            "summary": b3_summary,
        },
        "diagnosis": {
            "cause_code": cause_code,
            "cause_text": cause_text,
            "decision": "B2/B3 sensitivity determined from adaptive_ratio/fallback_count/mean_zlines.",
        },
    }

    with open(diagnosis_path, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)
    print("Saved:", diagnosis_path)

    # Patch primary results.json with diagnosis fields required by T3 task.
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            base = json.load(f)
    else:
        base = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(samples),
            "experiments": {},
        }

    base["diagnosis_t3"] = {
        "source": "experiments/ablation_adaptive_val/diagnosis_t3.json",
        "detector_params": diagnosis["detector_params"],
        "b2_summary": b2_summary,
        "b3_summary": b3_summary,
        "cause_code": cause_code,
        "cause_text": cause_text,
        "required_fields": [
            "adaptive_ratio",
            "fallback_count",
            "mean_zlines",
        ],
        "updated_at": datetime.now().isoformat(),
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2, ensure_ascii=False)
    print("Patched:", results_path)
    print("=" * 70)


if __name__ == "__main__":
    main()

