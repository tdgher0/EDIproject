"""
Perspective geometry utilities for EDIproject calibration.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def detect_line_segments(image: np.ndarray) -> np.ndarray:
    """
    Detect line segments using Canny edges and probabilistic Hough transform.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("image must be a 2D grayscale or 3D BGR image")

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10,
    )

    if lines is None:
        return np.empty((0, 4), dtype=np.float32)

    return lines.reshape(-1, 4).astype(np.float32)


def line_intersection(
    line1: np.ndarray,
    line2: np.ndarray,
) -> Optional[tuple[float, float]]:
    """
    Calculate the intersection of two infinite lines.

    Returns None when the lines are parallel.
    """
    x1, y1, x2, y2 = map(float, line1)
    x3, y3, x4, y4 = map(float, line2)

    denominator = (
        (x1 - x2) * (y3 - y4)
        - (y1 - y2) * (x3 - x4)
    )

    if abs(denominator) < 1e-10:
        return None

    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator

    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator

    return px, py


def estimate_vanishing_point(
    lines: np.ndarray,
) -> Optional[tuple[float, float]]:
    """
    Estimate one dominant vanishing point from line intersections.
    """
    if len(lines) < 2:
        return None

    intersections = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            point = line_intersection(lines[i], lines[j])

            if point is not None and np.all(np.isfinite(point)):
                intersections.append(point)

    if not intersections:
        return None

    points = np.asarray(intersections, dtype=np.float64)

    return (
        float(np.median(points[:, 0])),
        float(np.median(points[:, 1])),
    )


def estimate_vanishing_points(
    lines: np.ndarray,
) -> list[tuple[float, float]]:
    """
    Estimate dominant vanishing point information.

    Currently returns one dominant vanishing point.
    """
    point = estimate_vanishing_point(lines)

    if point is None:
        return []

    return [point]
