"""
Pixel-area calculation utilities for EDIproject.
"""

from __future__ import annotations


def pixel_area_to_square_meters(
    pixel_area: int | float,
    meters_per_pixel: float,
) -> float:
    """
    Convert an image area from pixels to square metres.

    Args:
        pixel_area: Area measured in pixels.
        meters_per_pixel: Physical scale in metres per pixel.

    Returns:
        Estimated area in square metres.

    Raises:
        ValueError: If either input is zero or negative.
    """
    if pixel_area <= 0:
        raise ValueError("pixel_area must be positive")

    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive")

    return pixel_area * (meters_per_pixel ** 2)