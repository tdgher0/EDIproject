"""
Wall-dimension measurement utilities for EDIproject.

The module estimates physical wall width and height from a metric
depth map, camera intrinsics, and an optional gravity-alignment
rotation matrix (Rtilt).

The module is independent of wall_area.py.
"""

from __future__ import annotations

import numpy as np


def _validate_inputs(
    wall_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rtilt: np.ndarray | None,
) -> None:
    """Validate inputs required for 3D wall-dimension calculation."""
    if not isinstance(wall_mask, np.ndarray):
        raise TypeError("wall_mask must be a NumPy array")

    if not isinstance(depth_map, np.ndarray):
        raise TypeError("depth_map must be a NumPy array")

    if wall_mask.ndim != 2:
        raise ValueError("wall_mask must be a 2D array")

    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a 2D array")

    if wall_mask.shape != depth_map.shape:
        raise ValueError("wall_mask and depth_map must have the same shape")

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

    if rtilt is not None:
        if not isinstance(rtilt, np.ndarray):
            raise TypeError("rtilt must be a NumPy array")

        if rtilt.shape != (3, 3):
            raise ValueError("rtilt must have shape (3, 3)")

        if not np.all(np.isfinite(rtilt)):
            raise ValueError("rtilt contains invalid values")


def backproject_wall_points(
    wall_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rtilt: np.ndarray | None = None,
) -> np.ndarray:
    """
    Back-project wall pixels into 3D metric coordinates.

    Args:
        wall_mask: Binary wall mask.
        depth_map: Metric depth map in metres.
        fx, fy: Camera focal lengths in pixels.
        cx, cy: Camera principal point.
        rtilt: Optional 3x3 gravity-alignment rotation matrix.

    Returns:
        Array of shape (N, 3), containing 3D wall points in metres.
    """
    _validate_inputs(
        wall_mask,
        depth_map,
        fx,
        fy,
        cx,
        cy,
        rtilt,
    )

    wall = wall_mask > 0

    if not np.any(wall):
        raise ValueError("wall_mask contains no wall pixels")

    y_coords, x_coords = np.nonzero(wall)
    z = depth_map[y_coords, x_coords].astype(np.float64)

    x = (x_coords.astype(np.float64) - cx) * z / fx
    y = (y_coords.astype(np.float64) - cy) * z / fy

    points = np.column_stack((x, y, z))

    if rtilt is not None:
        points = points @ rtilt.T

    return points.astype(np.float32)


def calculate_wall_width(
    wall_points: np.ndarray,
) -> float:
    """
    Calculate wall width from 3D wall points.

    Width is the horizontal X extent of the points.
    """
    if not isinstance(wall_points, np.ndarray):
        raise TypeError("wall_points must be a NumPy array")

    if wall_points.ndim != 2 or wall_points.shape[1] != 3:
        raise ValueError("wall_points must have shape (N, 3)")

    if wall_points.shape[0] < 2:
        raise ValueError("At least two wall points are required")

    if not np.all(np.isfinite(wall_points)):
        raise ValueError("wall_points contains invalid values")

    width = np.max(wall_points[:, 0]) - np.min(wall_points[:, 0])

    if width <= 0:
        raise ValueError("Unable to calculate a positive wall width")

    return float(width)


def calculate_wall_height(
    wall_points: np.ndarray,
) -> float:
    """
    Calculate wall height from 3D wall points.

    Height is the vertical Y extent of the points.
    """
    if not isinstance(wall_points, np.ndarray):
        raise TypeError("wall_points must be a NumPy array")

    if wall_points.ndim != 2 or wall_points.shape[1] != 3:
        raise ValueError("wall_points must have shape (N, 3)")

    if wall_points.shape[0] < 2:
        raise ValueError("At least two wall points are required")

    if not np.all(np.isfinite(wall_points)):
        raise ValueError("wall_points contains invalid values")

    height = np.max(wall_points[:, 1]) - np.min(wall_points[:, 1])

    if height <= 0:
        raise ValueError("Unable to calculate a positive wall height")

    return float(height)


def calculate_wall_dimensions(
    wall_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rtilt: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Calculate physical wall width and height.

    Returns:
        (width_m, height_m)
    """
    wall_points = backproject_wall_points(
        wall_mask=wall_mask,
        depth_map=depth_map,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rtilt=rtilt,
    )

    width = calculate_wall_width(wall_points)
    height = calculate_wall_height(wall_points)

    return width, height
