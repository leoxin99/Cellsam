"""
Detection parameter profiles.

This module centralizes the locked detection parameters for this project.
Only the `locked_eval` profile is active — it contains the best parameters
from E34/E34b (DAPI) and T3b (Adaptive) ablation experiments.

Historical runtime_default values are documented in CLAUDE.md for reference:
  DAPI: min_area=200, max_area=10000, edge=32, ratio=3.0, merge=1.2
  Adaptive: radius=256, min_zlines=15, zline_threshold=0.03

Goal:
- Single source of truth for all detection parameters
- Make active detection parameters explicit and auditable
"""

from copy import deepcopy
import json


DETECTION_PROFILES = {
    "locked_eval": {
        "description": "Frozen E34/E34b (DAPI) + T3b (Adaptive) parameters.",
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
            "search_radius": 160,        # T3b best (was 200 in E34)
            "min_zlines": 5,
            "zline_threshold": 0.05,     # T3b best (was 0.01 in E34)
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
