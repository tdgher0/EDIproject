from __future__ import annotations

from typing import Iterable

import numpy as np


def get_object_wall_overlap(
    wall_mask: np.ndarray,
    detections: Iterable[dict],
) -> np.ndarray:
    """
    Return the pixels where detected objects overlap the wall mask.

    Args:
        wall_mask: Boolean wall mask with shape (H, W).
        detections: Iterable of YOLO detection dictionaries. Each detection
            must contain a boolean "mask" array with shape (H, W).

    Returns:
        Boolean mask containing object pixels that lie inside the wall mask.
    """
    if not isinstance(wall_mask, np.ndarray):
        raise TypeError("wall_mask must be a NumPy array")

    if wall_mask.ndim != 2:
        raise ValueError("wall_mask must have shape (H, W)")

    wall_mask = wall_mask.astype(bool)

    object_wall_mask = np.zeros_like(wall_mask, dtype=bool)

    for detection in detections:
        if "mask" not in detection:
            raise KeyError("Each detection must contain a 'mask'")

        object_mask = np.asarray(detection["mask"])

        if object_mask.shape != wall_mask.shape:
            raise ValueError(
                "Detection mask shape must match wall_mask shape"
            )

        object_wall_mask |= object_mask.astype(bool) & wall_mask

    return object_wall_mask


def remove_objects_from_wall(
    wall_mask: np.ndarray,
    detections: Iterable[dict],
) -> np.ndarray:
    """
    Remove detected object pixels from a wall mask.

    Only object pixels that overlap the wall are removed.

    Args:
        wall_mask: Boolean wall mask with shape (H, W).
        detections: Iterable of YOLO detection dictionaries.

    Returns:
        Boolean wall mask with overlapping object pixels removed.
    """
    wall_mask = np.asarray(wall_mask).astype(bool)

    object_wall_mask = get_object_wall_overlap(
        wall_mask,
        detections,
    )

    return wall_mask & ~object_wall_mask
