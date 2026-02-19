"""
Detection module for CellSAM.

Import key functions from dapi module:
    from detection import detect_and_create_boxes
"""
from .dapi import (
    detect_nuclei,
    merge_close_nuclei,
    create_bounding_boxes,
    detect_and_create_boxes,
    is_on_edge,
)
from .profiles import (
    available_detection_profiles,
    get_detection_profile,
    apply_overrides,
    format_detection_profile_snapshot,
)

__all__ = [
    'detect_nuclei',
    'merge_close_nuclei',
    'create_bounding_boxes',
    'detect_and_create_boxes',
    'is_on_edge',
    'available_detection_profiles',
    'get_detection_profile',
    'apply_overrides',
    'format_detection_profile_snapshot',
]
