#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adaptive (Z-line) detection parameter ablation on VAL split (71 samples).

Protocol:
1) B1 search_radius sweep on val.
2) B2 min_zlines sweep with best B1 radius.
3) B3 zline_threshold sweep with best B1/B2.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir / "src"))

from ablation_adaptive_params import (
    evaluate_adaptive_params,
    load_test_samples,
)
from detection.profiles import (
    available_detection_profiles,
    get_detection_profile,
    apply_overrides,
    format_detection_profile_snapshot,
)


def main():
    def parse_int_csv(raw: str, arg_name: str):
        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(
                    "Invalid integer '{}' in {}={}".format(token, arg_name, raw)
                ) from exc
        if not values:
            raise ValueError("{} cannot be empty".format(arg_name))
        return values

    parser = argparse.ArgumentParser(description="Adaptive val ablation with checkpointable stages")
    parser.add_argument(
        "--stage",
        choices=["all", "b1", "b2", "b3"],
        default="all",
        help="Run all stages or a single stage.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing experiments/ablation_adaptive_val/results.json",
    )
    parser.add_argument(
        "--profile",
        choices=available_detection_profiles(),
        default="locked_eval",
        help="Detection profile source for min/max nucleus defaults.",
    )
    parser.add_argument(
        "--min-nucleus-area",
        type=int,
        default=None,
        help="Override min_nucleus_area passed to detect_with_adaptive_box.",
    )
    parser.add_argument(
        "--max-nucleus-area",
        type=int,
        default=None,
        help="Override max_nucleus_area passed to detect_with_adaptive_box.",
    )
    parser.add_argument(
        "--b1-values",
        type=str,
        default="200,300,400,500,600",
        help="Comma-separated search_radius candidates for B1.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/ablation_adaptive_val",
        help="Output directory for results.json.",
    )
    args = parser.parse_args()
    b1_values = parse_int_csv(args.b1_values, "--b1-values")
    profile_cfg = get_detection_profile(args.profile)
    adaptive_params = apply_overrides(
        profile_cfg["adaptive"],
        {
            "min_nucleus_area": args.min_nucleus_area,
            "max_nucleus_area": args.max_nucleus_area,
        },
    )
    requested_detector_params = {
        "min_nucleus_area": adaptive_params["min_nucleus_area"],
        "max_nucleus_area": adaptive_params["max_nucleus_area"],
    }

    print("=" * 70)
    print("Adaptive (Z-line) standardized ablation (Val set - 71 samples)")
    print("=" * 70)

    data_dir = project_dir / "data" / "processed"
    split_file = project_dir / "data" / "splits" / "val_ids.txt"
    output_dir_arg = Path(args.output_dir)
    output_dir = output_dir_arg if output_dir_arg.is_absolute() else project_dir / output_dir_arg
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "results.json"

    print("\n[1] Loading VAL samples...")
    samples = load_test_samples(data_dir, split_file, n_samples=71)
    print("    Loaded {} samples".format(len(samples)))
    print("    Detection profile snapshot:")
    print(format_detection_profile_snapshot(args.profile, profile_cfg["dapi"], adaptive_params))

    if args.resume and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print("    Resume from {}".format(output_file))
        existing_profile = results.get("detection_profile")
        if existing_profile and existing_profile != args.profile:
            raise RuntimeError(
                "detection_profile mismatch under --resume. "
                "Existing={} Requested={}.".format(existing_profile, args.profile)
            )
        existing_detector_params = results.get("detector_params")
        if existing_detector_params and existing_detector_params != requested_detector_params:
            raise RuntimeError(
                "detector_params mismatch under --resume. "
                "Existing={} Requested={}. "
                "Please rerun with matching params, or remove results.json and restart from --stage b1.".format(
                    existing_detector_params, requested_detector_params
                )
            )
        results["detector_params"] = requested_detector_params
    else:
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(samples),
            "detection_profile": args.profile,
            "detector_params": requested_detector_params,
            "experiments": {},
        }
    results["detection_profile"] = args.profile
    print("    Detector params:", results["detector_params"])

    default_min_zlines = 15
    default_zline_threshold = 0.03

    # Sweep value lists — change here to update the search space
    b2_values = [5, 10, 15, 20, 30]
    b3_values = [0.01, 0.02, 0.03, 0.05, 0.1]

    import hashlib
    sweep_signature = hashlib.md5(
        json.dumps({"b1": b1_values, "b2": b2_values, "b3": b3_values}).encode()
    ).hexdigest()[:8]

    if "sweep_signature" not in results:
        results["sweep_signature"] = sweep_signature
    elif results["sweep_signature"] != sweep_signature:
        raise RuntimeError(
            "sweep_signature mismatch under --resume. "
            "Existing={} Current={}. "
            "Sweep value lists have changed; remove results.json and restart.".format(
                results["sweep_signature"], sweep_signature
            )
        )
    print("    Sweep signature:", sweep_signature)

    def save_results():
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("    Saved checkpoint:", output_file)

    if args.stage in ("all", "b1"):
        print("\n[B1] search_radius sweep...")
        b1_results = []
        for value in b1_values:
            metrics = evaluate_adaptive_params(
                samples,
                search_radius=value,
                min_zlines=default_min_zlines,
                zline_threshold=default_zline_threshold,
                min_nucleus_area=requested_detector_params["min_nucleus_area"],
                max_nucleus_area=requested_detector_params["max_nucleus_area"],
            )
            metrics["value"] = value
            b1_results.append(metrics)
            print(
                "    search_radius={}: F1={:.4f}, P={:.4f}, R={:.4f}, AR={:.3f}, FB={}".format(
                    value,
                    metrics["f1"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics.get("adaptive_ratio", 0.0),
                    metrics.get("fallback_count", 0),
                )
            )

        best_b1 = max(b1_results, key=lambda x: x["f1"])
        results["experiments"]["B1_search_radius"] = {
            "param": "search_radius",
            "results": b1_results,
            "best_value": best_b1["value"],
            "best_f1": best_b1["f1"],
        }
        save_results()

    b1_data = results["experiments"].get("B1_search_radius")
    if b1_data is None:
        raise RuntimeError("B1 result missing. Run --stage b1 first, or use --stage all.")
    best_b1_value = b1_data["best_value"]

    if args.stage in ("all", "b2"):
        print("\n[B2] min_zlines sweep (search_radius={})...".format(best_b1_value))
        b2_results = []
        for value in b2_values:
            metrics = evaluate_adaptive_params(
                samples,
                search_radius=best_b1_value,
                min_zlines=value,
                zline_threshold=default_zline_threshold,
                min_nucleus_area=requested_detector_params["min_nucleus_area"],
                max_nucleus_area=requested_detector_params["max_nucleus_area"],
            )
            metrics["value"] = value
            b2_results.append(metrics)
            print(
                "    min_zlines={}: F1={:.4f}, P={:.4f}, R={:.4f}, AR={:.3f}, FB={}".format(
                    value,
                    metrics["f1"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics.get("adaptive_ratio", 0.0),
                    metrics.get("fallback_count", 0),
                )
            )

        best_b2 = max(b2_results, key=lambda x: x["f1"])
        results["experiments"]["B2_min_zlines"] = {
            "param": "min_zlines",
            "fixed": {
                "search_radius": best_b1_value,
                "zline_threshold": default_zline_threshold,
            },
            "results": b2_results,
            "best_value": best_b2["value"],
            "best_f1": best_b2["f1"],
        }
        save_results()

    b2_data = results["experiments"].get("B2_min_zlines")
    if b2_data is None and args.stage in ("all", "b3"):
        raise RuntimeError("B2 result missing. Run --stage b2 first, or use --stage all.")

    if args.stage in ("all", "b3"):
        best_b2_value = b2_data["best_value"]
        print(
            "\n[B3] zline_threshold sweep (search_radius={}, min_zlines={})...".format(
                best_b1_value, best_b2_value
            )
        )
        b3_results = []
        for value in b3_values:
            metrics = evaluate_adaptive_params(
                samples,
                search_radius=best_b1_value,
                min_zlines=best_b2_value,
                zline_threshold=value,
                min_nucleus_area=requested_detector_params["min_nucleus_area"],
                max_nucleus_area=requested_detector_params["max_nucleus_area"],
            )
            metrics["value"] = value
            b3_results.append(metrics)
            print(
                "    zline_threshold={}: F1={:.4f}, P={:.4f}, R={:.4f}, AR={:.3f}, FB={}".format(
                    value,
                    metrics["f1"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics.get("adaptive_ratio", 0.0),
                    metrics.get("fallback_count", 0),
                )
            )

        best_b3 = max(b3_results, key=lambda x: x["f1"])
        results["experiments"]["B3_zline_threshold"] = {
            "param": "zline_threshold",
            "fixed": {
                "search_radius": best_b1_value,
                "min_zlines": best_b2_value,
            },
            "results": b3_results,
            "best_value": best_b3["value"],
            "best_f1": best_b3["f1"],
        }
        save_results()

    b1_data = results["experiments"].get("B1_search_radius")
    b2_data = results["experiments"].get("B2_min_zlines")
    b3_data = results["experiments"].get("B3_zline_threshold")
    if b1_data and b2_data and b3_data:
        final_optimal = {
            "search_radius": b1_data["best_value"],
            "min_zlines": b2_data["best_value"],
            "zline_threshold": b3_data["best_value"],
            "f1": b3_data["best_f1"],
        }
        results["final_optimal"] = final_optimal
        save_results()

    print("\n" + "=" * 70)
    print("Standardized result summary (Val set)")
    print("=" * 70)
    print("{:<25} {:>15} {:>10}".format("Experiment", "Best Value", "Best F1"))
    print("-" * 55)
    if b1_data:
        print(
            "{:<25} {:>15} {:>10.4f}".format(
                "B1_search_radius", str(b1_data["best_value"]), b1_data["best_f1"]
            )
        )
    if b2_data:
        print(
            "{:<25} {:>15} {:>10.4f}".format(
                "B2_min_zlines", str(b2_data["best_value"]), b2_data["best_f1"]
            )
        )
    if b3_data:
        print(
            "{:<25} {:>15} {:>10.4f}".format(
                "B3_zline_threshold", str(b3_data["best_value"]), b3_data["best_f1"]
            )
        )
    print("-" * 55)
    if "final_optimal" in results:
        combo = "{}/{}/{}".format(
            results["final_optimal"]["search_radius"],
            results["final_optimal"]["min_zlines"],
            results["final_optimal"]["zline_threshold"],
        )
        print(
            "{:<25} {:>15} {:>10.4f}".format(
                "Final optimal", combo, results["final_optimal"]["f1"]
            )
        )
    else:
        print("{:<25} {:>15} {:>10}".format("Final optimal", "not ready", "-"))

    # Sensitivity warning for diagnosis
    b2_rows = b2_data["results"] if b2_data else []
    b3_rows = b3_data["results"] if b3_data else []
    if b2_rows:
        b2_f1 = [row["f1"] for row in b2_rows]
        if max(b2_f1) - min(b2_f1) < 1e-4:
            print("Warning: B2 appears insensitive on this setting (flat F1 across min_zlines).")
    if b3_rows:
        b3_f1 = [row["f1"] for row in b3_rows]
        if max(b3_f1) - min(b3_f1) < 1e-4:
            print("Warning: B3 appears insensitive on this setting (flat F1 across zline_threshold).")

    print("\nSaved:", output_file)
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
