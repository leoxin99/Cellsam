"""
Detection parameter profiles.

This module centralizes the two execution profiles used in this project:
1) runtime_default: follow detection function defaults in src/detection/dapi.py
2) locked_eval: frozen parameters for standardized val/test evaluation

Goal:
- Reduce accidental use of runtime defaults in final evaluation
- Make active detection parameters explicit and auditable
"""

from copy import deepcopy
import json


DETECTION_PROFILES = {
    "runtime_default": {
        "description": "Code defaults in src/detection/dapi.py (for daily runtime).",
        "dapi": {
            "min_nucleus_area": 200,
            "max_nucleus_area": 10000,
            "edge_margin": 32,
            "size_ratio_threshold": 3.0,
            "merge_coeff": 1.2,
            "use_relative_distance": True,
            "fixed_merge_distance": 373,
        },
        "adaptive": {
            "min_nucleus_area": 200,
            "max_nucleus_area": 10000,
            "search_radius": 256,
            "min_zlines": 15,
            "zline_threshold": 0.03,
            "edge_margin": 32,
            "size_ratio_threshold": 3.0,
            "merge_coeff": 1.2,
            "use_relative_distance": True,
            "fixed_merge_distance": 373,
        },
    },
    "locked_eval": {
        "description": "Frozen E34/E34b eval parameters (for standardized reporting).",
        "dapi": {
            "min_nucleus_area": 1500,
            "max_nucleus_area": 20000,
            "edge_margin": 20,
            "size_ratio_threshold": 2.5,
            "merge_coeff": 1.4,
            "use_relative_distance": True,
            "fixed_merge_distance": 373,
        },
        "adaptive": {
            "min_nucleus_area": 1500,
            "max_nucleus_area": 20000,
            "search_radius": 200,
            "min_zlines": 5,
            "zline_threshold": 0.01,
            "edge_margin": 20,
            "size_ratio_threshold": 2.5,
            "merge_coeff": 1.4,
            "use_relative_distance": True,
            "fixed_merge_distance": 373,
        },
    },
}


def available_detection_profiles():
    """Return all valid profile names."""
    return tuple(DETECTION_PROFILES.keys())


def get_detection_profile(profile_name: str):
    """Get a deep-copied profile payload by name."""
    if profile_name not in DETECTION_PROFILES:
        raise ValueError(
            f"Unknown detection profile: {profile_name}. "
            f"Available: {', '.join(available_detection_profiles())}"
        )
    return deepcopy(DETECTION_PROFILES[profile_name])


def apply_overrides(base_params: dict, overrides: dict):
    """Apply non-None overrides onto a base param dict."""
    result = deepcopy(base_params)
    if not overrides:
        return result
    for key, value in overrides.items():
        if value is None:
            continue
        if key in result:
            result[key] = value
    return result


def format_detection_profile_snapshot(
    profile_name: str,
    dapi_params: dict,
    adaptive_params: dict = None,
):
    """Render profile snapshot as pretty JSON text for logs."""
    payload = {
        "profile": profile_name,
        "dapi": dapi_params,
    }
    if adaptive_params is not None:
        payload["adaptive"] = adaptive_params
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
