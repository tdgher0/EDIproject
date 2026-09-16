from __future__ import annotations

import numpy as np

from src.measurement.wall_area import _triangular_mesh_area


def _validate_inputs(
    opening_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    if not isinstance(opening_mask, np.ndarray):
        raise TypeError("opening_mask must be a NumPy array")

    if not isinstance(depth_map, np.ndarray):
        raise TypeError("depth_map must be a NumPy array")

    if opening_mask.ndim != 2:
        raise ValueError("opening_mask must be a 2D array")

    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a 2D array")

    if opening_mask.shape != depth_map.shape:
        raise ValueError(
            "opening_mask and depth_map must have the same shape"
        )

    if not np.isfinite(fx) or fx <= 0:
        raise ValueError("fx must be positive and finite")

    if not np.isfinite(fy) or fy <= 0:
        raise ValueError("fy must be positive and finite")

    if not np.isfinite(cx):
        raise ValueError("cx must be finite")

    if not np.isfinite(cy):
        raise ValueError("cy must be finite")

    if not np.all(np.isfinite(depth_map)):
        raise ValueError("depth_map contains invalid values")

    if np.any(depth_map <= 0):
        raise ValueError("depth_map must contain positive metric depths")


def calculate_opening_area_from_depth(
    opening_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth_jump: float | None = 0.025,
) -> float:
    """Same 3D mesh as wall area. Default max_depth_jump matches production (0.025 m)."""
    _validate_inputs(
        opening_mask,
        depth_map,
        fx,
        fy,
        cx,
        cy,
    )

    if max_depth_jump is not None and max_depth_jump < 0:
        raise ValueError("max_depth_jump must be non-negative")

    if not np.any(opening_mask > 0):
        raise ValueError("opening_mask contains no opening pixels")

    area = _triangular_mesh_area(
        opening_mask,
        depth_map,
        fx,
        fy,
        cx,
        cy,
        max_depth_jump,
    )
    if area <= 0:
        raise ValueError(
            "Unable to calculate a positive opening area"
        )
    return area
