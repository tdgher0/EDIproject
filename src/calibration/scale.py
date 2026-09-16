"""
Calibration helpers for converting MiDaS relative depth to metric depth.

This module is deliberately lightweight and does not perform any
image‑specific processing.  It only contains two public helpers that
build on the reference‑object based calibration already implemented in
``src.calibration.reference_object``.

The typical use‑case is:

1.  A reference object (e.g. a standard 1.75 m tall person or a
    known‑height object) is detected in the image.  Its pixel height
    and the MiDaS depth value at that pixel are known.
2.  ``compute_calibration_factor`` is called to obtain the scalar that
    converts MiDaS *relative* depth into metres.
3.  The whole depth map is then scaled by that factor using
    ``apply_calibration``.

Both functions perform basic validation and provide clear error
messages if the inputs are not as expected.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# Import the two reference‑object utilities.  Importing at module
# level keeps the public API simple while avoiding circular imports.
from src.calibration.reference_object import (
    metric_depth_from_midas,
    reference_depth_from_height,
)

__all__: Final[list[str]] = [
    "compute_calibration_factor",
    "apply_calibration",
]


def compute_calibration_factor(
    reference_height_m: float,
    reference_pixel_height: int,
    camera_fy: float,
    midas_rel_depth_at_ref: float,
) -> float:
    """Compute a scalar that converts MiDaS relative depth to metres.

    Parameters
    ----------
    reference_height_m:
        The real‑world height of the reference object in metres.
    reference_pixel_height:
        Height of the reference in the image, measured in pixels.
    camera_fy:
        The vertical focal length of the camera in pixel units.
    midas_rel_depth_at_ref:
        The MiDaS relative depth value at the centre of the reference
        object.

    Returns
    -------
    float
        The calibration factor ``k`` such that
        ``metric_depth = k / midas_relative_depth``.

    Notes
    -----
    The factor is derived as::

        k = reference_depth * midas_rel_depth_at_ref

    where ``reference_depth`` is the metric depth of the reference
    object obtained from its known height and the camera intrinsics.
    """
    if reference_height_m <= 0:
        raise ValueError("reference_height_m must be positive")
    if reference_pixel_height <= 0:
        raise ValueError("reference_pixel_height must be positive")
    if camera_fy <= 0:
        raise ValueError("camera_fy must be positive")
    if midas_rel_depth_at_ref <= 0:
        raise ValueError("midas_rel_depth_at_ref must be positive")

    ref_depth_m = reference_depth_from_height(
        reference_height_m, reference_pixel_height, camera_fy
    )

    # Calibration scale (inverse depth model):
    #    scale_a = reference_depth_m * midas_rel_depth_at_ref
    #    metric_depth = scale_a / midas_rel_depth
    scale_a = ref_depth_m * midas_rel_depth_at_ref

    return scale_a


def apply_calibration(
    midas_relative_depth_map: np.ndarray, calibration_factor: float
) -> np.ndarray:
    """Scale a full MiDaS relative‑depth map to metric depth.

    Parameters
    ----------
    midas_relative_depth_map:
        2‑D array of MiDaS relative depth values.
    calibration_factor:
        Scalar returned by :func:`compute_calibration_factor`.

    Returns
    -------
    numpy.ndarray
        Metric depth map of the same shape in metres.

    Raises
    ------
    TypeError
        If the inputs are not ``numpy.ndarray`` or the factor is not a
        numeric scalar.
    ValueError
        If the depth map is empty or the factor is non‑positive.
    """
    if not isinstance(midas_relative_depth_map, np.ndarray):
        raise TypeError(
            f"midas_relative_depth_map must be a numpy.ndarray, got {type(midas_relative_depth_map).__name__}"
        )
    if midas_relative_depth_map.ndim != 2:
        raise ValueError(
            f"midas_relative_depth_map must be 2‑D, got {midas_relative_depth_map.ndim}‑D array"
        )
    if midas_relative_depth_map.size == 0:
        raise ValueError("midas_relative_depth_map cannot be empty")

    if not isinstance(calibration_factor, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"calibration_factor must be numeric, got {type(calibration_factor).__name__}"
        )
    if not np.isfinite(calibration_factor) or calibration_factor <= 0:
        raise ValueError("calibration_factor must be positive and finite")

    # Delegate conversion to metric_depth_from_midas using the calibration factor.
    return metric_depth_from_midas(midas_relative_depth_map, calibration_factor)

# End of file
