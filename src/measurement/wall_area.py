from __future__ import annotations

import numpy as np


def calculate_wall_pixel_area(
    wall_mask: np.ndarray,
) -> int:
    if wall_mask.ndim != 2:
        raise ValueError("wall_mask must be a 2D array")

    wall_pixels = np.count_nonzero(wall_mask)
    if wall_pixels <= 0:
        raise ValueError("wall_mask contains no wall pixels")

    return int(wall_pixels)


def calculate_baseline_wall_area(
    wall_mask: np.ndarray,
    meters_per_pixel: float,
) -> float:
    """Uniform pixel scale A_real = A_pixel × S². Ignores perspective and depth."""
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive")

    pixel_area = calculate_wall_pixel_area(wall_mask)
    return pixel_area * (meters_per_pixel ** 2)


def _validate_depth_inputs(
    wall_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    if wall_mask.ndim != 2:
        raise ValueError("wall_mask must be a 2D array")

    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a 2D array")

    if wall_mask.shape != depth_map.shape:
        raise ValueError(
            "wall_mask and depth_map must have the same shape"
        )

    if fx <= 0:
        raise ValueError("fx must be positive")

    if fy <= 0:
        raise ValueError("fy must be positive")

    if not np.all(np.isfinite(depth_map)):
        raise ValueError("depth_map contains invalid values")

    if np.any(depth_map <= 0):
        raise ValueError("depth_map must contain positive metric depths")

    if not np.isfinite(cx):
        raise ValueError("cx must be finite")

    if not np.isfinite(cy):
        raise ValueError("cy must be finite")


def _triangular_mesh_area(
    region_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth_jump: float | None,
) -> float:
    """Pinhole backprojection and two-triangle mesh area.

    X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy
    A = 0.5 * ||(P2 - P1) × (P3 - P1)||
    Depth-jump uses the original metric Z values, not the float32 copies used
    for backprojection.
    """
    region = region_mask > 0
    height, width = depth_map.shape

    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    z = depth_map.astype(np.float32)
    x = (x_coords - cx) * z / fx
    y = (y_coords - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)

    p00 = region[:-1, :-1]
    p01 = region[:-1, 1:]
    p10 = region[1:, :-1]
    p11 = region[1:, 1:]

    pts00 = points[:-1, :-1]
    pts01 = points[:-1, 1:]
    pts10 = points[1:, :-1]
    pts11 = points[1:, 1:]

    z00 = depth_map[:-1, :-1]
    z01 = depth_map[:-1, 1:]
    z10 = depth_map[1:, :-1]
    z11 = depth_map[1:, 1:]

    tri1 = p00 & p01 & p10
    tri2 = p01 & p10 & p11
    if max_depth_jump is not None:
        jump1 = np.maximum(np.maximum(z00, z01), z10) - np.minimum(
            np.minimum(z00, z01), z10
        )
        jump2 = np.maximum(np.maximum(z01, z10), z11) - np.minimum(
            np.minimum(z01, z10), z11
        )
        tri1 = tri1 & (jump1 <= max_depth_jump)
        tri2 = tri2 & (jump2 <= max_depth_jump)

    area1 = 0.5 * np.linalg.norm(np.cross(pts01 - pts00, pts10 - pts00), axis=-1)
    area2 = 0.5 * np.linalg.norm(np.cross(pts10 - pts01, pts11 - pts01), axis=-1)
    return float(area1[tri1].sum() + area2[tri2].sum())


def calculate_wall_area_from_depth(
    wall_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth_jump: float | None = None,
) -> float:
    """Metric-depth wall area. MiDaS relative depth must be calibrated first."""
    _validate_depth_inputs(
        wall_mask,
        depth_map,
        fx,
        fy,
        cx,
        cy,
    )

    if max_depth_jump is not None and max_depth_jump < 0:
        raise ValueError("max_depth_jump must be non-negative")

    if not np.any(wall_mask > 0):
        raise ValueError("wall_mask contains no wall pixels")

    area = _triangular_mesh_area(
        wall_mask,
        depth_map,
        fx,
        fy,
        cx,
        cy,
        max_depth_jump,
    )
    if area <= 0:
        raise ValueError(
            "Unable to calculate a positive wall surface area"
        )
    return area
