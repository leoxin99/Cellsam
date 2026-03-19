"""H1bA candidate extraction and query-prior preparation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .dapi import (
    create_adaptive_box,
    create_bounding_boxes,
    detect_nuclei,
    detect_zlines_in_region,
    filter_by_actn2,
    get_cell_zlines,
    is_on_edge,
    merge_close_nuclei,
)
from .profiles import get_detection_profile


Candidate = Dict[str, object]

_DEFAULT_ACTN2_COVERAGE_THRESHOLD = 0.3


def _to_chw(raw_image: np.ndarray) -> np.ndarray:
    """Convert input image to (C, H, W) layout."""
    if raw_image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={raw_image.shape}")

    if raw_image.shape[0] <= 8 and raw_image.shape[-1] > 8:
        return raw_image
    if raw_image.shape[-1] <= 8 and raw_image.shape[0] > 8:
        return raw_image.transpose(2, 0, 1)
    if raw_image.shape[0] in (3, 4, 5):
        return raw_image
    if raw_image.shape[-1] in (3, 4, 5):
        return raw_image.transpose(2, 0, 1)

    raise ValueError(
        f"Expected at least 3 channels for BF/DAPI/Actn2, got shape={raw_image.shape}"
    )


def _to_uint8(channel: np.ndarray) -> np.ndarray:
    """Convert arbitrary numeric channel data to uint8."""
    channel = np.asarray(channel)
    if channel.size == 0:
        return channel.astype(np.uint8)

    if channel.dtype == np.uint8:
        return channel

    channel = channel.astype(np.float32)
    channel_max = float(channel.max())
    channel_min = float(channel.min())

    if channel_max <= 1.0 and channel_min >= 0.0:
        return np.clip(channel * 255.0, 0, 255).astype(np.uint8)

    if channel_max > 255.0 or channel_min < 0.0:
        if channel_max <= channel_min:
            return np.zeros_like(channel, dtype=np.uint8)
        scaled = (channel - channel_min) / (channel_max - channel_min)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    return np.clip(channel, 0, 255).astype(np.uint8)


def _extract_dapi_actn2(raw_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract DAPI and Actn2 channels from a processed raw image."""
    image_chw = _to_chw(raw_image)
    dapi_channel = _to_uint8(image_chw[1])
    actn2_channel = _to_uint8(image_chw[2])
    return dapi_channel, actn2_channel


def _group_center_xy(group: list) -> Tuple[float, float]:
    """Compute the center of one merged nucleus group as (x, y)."""
    all_coords = np.concatenate([region.coords for region in group], axis=0)
    center_y = float(all_coords[:, 0].mean())
    center_x = float(all_coords[:, 1].mean())
    return center_x, center_y


def _group_box_xyxy(group: list, image_shape: Tuple[int, int], margin: int) -> List[float]:
    """Build one DAPI-style box for a merged nucleus group."""
    boxes = create_bounding_boxes(
        [group],
        image_shape,
        exclude_edges=False,
        margin=margin,
    )
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(value) for value in boxes[0]]


def _make_candidate(
    *,
    group: list,
    box_xyxy: List[float],
    source_mode: str,
    num_zlines: int,
    adaptive: bool,
) -> Candidate:
    center_xy = _group_center_xy(group)
    return {
        "center_xy": [float(center_xy[0]), float(center_xy[1])],
        "box_xyxy": [float(value) for value in box_xyxy],
        "source_mode": source_mode,
        "group_size": int(len(group)),
        "is_binuclear": bool(len(group) > 1),
        "num_zlines": int(num_zlines),
        "adaptive": bool(adaptive),
    }


def _detect_adaptive_candidates(
    dapi_channel: np.ndarray,
    actn2_channel: np.ndarray,
    profile_name: str,
    adaptive_box_mode: str = "center_only",
) -> Tuple[List[Candidate], Dict[str, int]]:
    """Detect adaptive candidates.

    Modes:
    - center_only (default): no Z-line geometry; candidate box comes from DAPI-only box.
    - adaptive_zline: keep legacy Z-line -> adaptive/fallback box behavior.
    """
    params = get_detection_profile(profile_name)["adaptive"]
    if adaptive_box_mode not in {"center_only", "adaptive_zline"}:
        raise ValueError(
            f"Unsupported adaptive_box_mode={adaptive_box_mode}. "
            "Expected 'center_only' or 'adaptive_zline'."
        )

    regions = detect_nuclei(
        dapi_channel,
        min_area=params["min_nucleus_area"],
        max_area=params["max_nucleus_area"],
    )
    cell_groups = merge_close_nuclei(
        regions,
        size_ratio_threshold=params["size_ratio_threshold"],
        use_relative_distance=params["use_relative_distance"],
        fixed_merge_distance=params["fixed_merge_distance"],
        merge_coeff=params["merge_coeff"],
    )

    image_shape = dapi_channel.shape
    candidates: List[Candidate] = []
    edge_filtered = 0

    for group in cell_groups:
        if any(is_on_edge(r, image_shape, params["edge_margin"]) for r in group):
            edge_filtered += 1
            continue

        if adaptive_box_mode == "adaptive_zline":
            center_x, center_y = _group_center_xy(group)
            zlines = detect_zlines_in_region(
                actn2_channel,
                int(round(center_y)),
                int(round(center_x)),
                search_radius=params["search_radius"],
                threshold=params["zline_threshold"],
            )
            cell_zlines = get_cell_zlines(
                zlines,
                (center_y, center_x),
                max_distance=params["search_radius"] * 0.8,
            )
            box_xyxy = create_adaptive_box(
                group,
                cell_zlines,
                image_shape,
                min_zlines=params["min_zlines"],
            )
            num_zlines = len(cell_zlines)
            adaptive_flag = len(cell_zlines) >= params["min_zlines"]
        else:
            # Center-only runtime: skip Z-line work entirely.
            box_xyxy = _group_box_xyxy(group, image_shape, params["edge_margin"])
            num_zlines = 0
            adaptive_flag = False

        candidates.append(
            _make_candidate(
                group=group,
                box_xyxy=box_xyxy,
                source_mode="adaptive",
                num_zlines=num_zlines,
                adaptive=adaptive_flag,
            )
        )

    stats = {
        "n_regions": int(len(regions)),
        "n_groups": int(len(cell_groups)),
        "n_edge_filtered": int(edge_filtered),
        "n_candidates": int(len(candidates)),
        "n_binuclear_candidates": int(sum(c["is_binuclear"] for c in candidates)),
        "n_adaptive_candidates": int(sum(c["adaptive"] for c in candidates)),
    }
    return candidates, stats


def _detect_dapi_cm_candidates(
    dapi_channel: np.ndarray,
    actn2_channel: np.ndarray,
    profile_name: str,
) -> Tuple[List[Candidate], Dict[str, int]]:
    params = get_detection_profile(profile_name)["dapi"]
    regions = detect_nuclei(
        dapi_channel,
        min_area=params["min_nucleus_area"],
        max_area=params["max_nucleus_area"],
    )
    cell_groups = merge_close_nuclei(
        regions,
        size_ratio_threshold=params["size_ratio_threshold"],
        use_relative_distance=params["use_relative_distance"],
        fixed_merge_distance=params["fixed_merge_distance"],
        merge_coeff=params["merge_coeff"],
    )

    if np.count_nonzero(actn2_channel) == 0:
        filtered_groups = []
    else:
        filtered_groups = filter_by_actn2(
            cell_groups,
            actn2_channel,
            coverage_threshold=_DEFAULT_ACTN2_COVERAGE_THRESHOLD,
        )

    image_shape = dapi_channel.shape
    candidates: List[Candidate] = []
    edge_filtered = 0

    for group in filtered_groups:
        if any(is_on_edge(r, image_shape, params["edge_margin"]) for r in group):
            edge_filtered += 1
            continue

        candidates.append(
            _make_candidate(
                group=group,
                box_xyxy=_group_box_xyxy(group, image_shape, params["edge_margin"]),
                source_mode="dapi_cm",
                num_zlines=0,
                adaptive=False,
            )
        )

    stats = {
        "n_regions": int(len(regions)),
        "n_groups": int(len(cell_groups)),
        "n_filtered_groups": int(len(filtered_groups)),
        "n_edge_filtered": int(edge_filtered),
        "n_candidates": int(len(candidates)),
        "n_binuclear_candidates": int(sum(c["is_binuclear"] for c in candidates)),
        "n_adaptive_candidates": 0,
    }
    return candidates, stats


def _detect_h1b_candidates_with_stats(
    raw_image: np.ndarray,
    profile_name: str = "locked_eval",
    candidate_mode: str = "adaptive",
    adaptive_box_mode: str = "center_only",
) -> Tuple[List[Candidate], Dict[str, int]]:
    dapi_channel, actn2_channel = _extract_dapi_actn2(raw_image)

    if candidate_mode == "adaptive":
        return _detect_adaptive_candidates(
            dapi_channel,
            actn2_channel,
            profile_name,
            adaptive_box_mode=adaptive_box_mode,
        )
    if candidate_mode == "dapi_cm":
        return _detect_dapi_cm_candidates(dapi_channel, actn2_channel, profile_name)

    raise ValueError(
        f"Unsupported candidate_mode={candidate_mode}. Expected 'adaptive' or 'dapi_cm'."
    )


def detect_h1b_candidates(
    raw_image: np.ndarray,
    profile_name: str = "locked_eval",
    candidate_mode: str = "adaptive",
    adaptive_box_mode: str = "center_only",
) -> List[Candidate]:
    """Run the frozen H1bA candidate pipeline on one image."""
    candidates, _ = _detect_h1b_candidates_with_stats(
        raw_image=raw_image,
        profile_name=profile_name,
        candidate_mode=candidate_mode,
        adaptive_box_mode=adaptive_box_mode,
    )
    return candidates


def candidates_to_query_priors(
    candidates: List[Candidate],
    image_shape: Tuple[int, int],
    max_queries: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert candidate dicts into padded normalized query priors."""
    if max_queries <= 0:
        raise ValueError(f"max_queries must be positive, got {max_queries}")

    image_h, image_w = image_shape
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"Invalid image_shape={image_shape}")

    points = torch.zeros((max_queries, 2), dtype=torch.float32)
    valid_mask = torch.zeros((max_queries,), dtype=torch.bool)
    boxes = torch.zeros((max_queries, 4), dtype=torch.float32)

    scale_points = torch.tensor(
        [max(image_w - 1, 1), max(image_h - 1, 1)],
        dtype=torch.float32,
    )
    scale_boxes = torch.tensor(
        [max(image_w, 1), max(image_h, 1), max(image_w, 1), max(image_h, 1)],
        dtype=torch.float32,
    )

    for index, candidate in enumerate(candidates[:max_queries]):
        center_xy = torch.tensor(candidate["center_xy"], dtype=torch.float32)
        box_xyxy = torch.tensor(candidate["box_xyxy"], dtype=torch.float32)
        points[index] = torch.clamp(center_xy / scale_points, 0.0, 1.0)
        boxes[index] = torch.clamp(box_xyxy / scale_boxes, 0.0, 1.0)
        valid_mask[index] = True

    return points, valid_mask, boxes


__all__ = [
    "detect_h1b_candidates",
    "candidates_to_query_priors",
]
