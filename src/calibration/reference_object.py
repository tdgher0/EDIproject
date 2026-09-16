"""
Calibration utilities for generating metric depth maps.

This module provides lightweight utilities to compute metric depth from a known reference height,
to fit a single‑scale MiDaS calibration, and to generate metric depth maps from MiDaS relative
depth predictions.

All calculations are performed with NumPy and the public API is intentionally free of any
dataset‑specific logic or OpenCV dependencies.
"""

import numpy as np

__all__ = [
    "reference_depth_from_height",
    "fit_midas_single_scale",
    "metric_depth_from_midas",
]


def _validate_positive_finite(value, name: str):
    """
    Internal helper that validates a numeric value is finite and > 0.

    Parameters
    ----------
    value : float or np.ndarray
        The value(s) to validate.
    name : str
        Name of the variable, used in the exception message.

    Raises
    ------
    ValueError
        If any element is non‑finite or not positive.
    """
    if isinstance(value, (float, int)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    else:
        if not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError(f"{name} must be positive and finite")


def reference_depth_from_height(reference_real_height_m: float,
                                reference_pixel_height: float,
                                fy: float) -> np.float32:
    """
    Compute the metric depth of a reference object given its real height and
    observed pixel height in the image.

    The formula derives from the pinhole camera model:

        Z = H * fy / h

    where
        H  = known real height (m)
        fy = focal length in pixel units (vertical)
        h  = pixel height of the reference object

    Parameters
    ----------
    reference_real_height_m : float
        Physical height of the reference object in metres.
    reference_pixel_height : float
        Height of the reference object in the image, in pixels.
    fy : float
        Vertical focal length of the camera in pixels.

    Returns
    -------
    np.float32
        The metric depth (m) of the reference object.

    Raises
    ------
    ValueError
        If any argument is not positive or not finite.
    """
    _validate_positive_finite(reference_real_height_m, "reference_real_height_m")
    _validate_positive_finite(reference_pixel_height, "reference_pixel_height")
    _validate_positive_finite(fy, "fy")

    depth = reference_real_height_m * fy / reference_pixel_height
    return np.float32(depth)



def fit_midas_single_scale(midas_value: np.ndarray,
                           metric_depth_m: np.ndarray) -> float:
    """
    Estimate a single‑scale MiDaS calibration factor ``a`` that maps MiDaS relative
    depth predictions to metric depth:

        Z_metric = a / D_midas

    The factor ``a`` is computed as the mean of ``metric_depth_m * midas_value`` over
    all valid pixels.  Pixels where the MiDaS value is <= 0 or not finite are ignored.

    Parameters
    ----------
    midas_value : np.ndarray
        MiDaS relative depth map.  Expected to be a float array with positive values.
    metric_depth_m : np.ndarray
        Ground‑truth metric depth map of the same shape.

    Returns
    -------
    float
        The fitted scale factor ``a``.

    Raises
    ------
    ValueError
        If inputs are not the same shape, or if no valid pixels exist.
    """
    if midas_value.shape != metric_depth_m.shape:
        raise ValueError("midas_value and metric_depth_m must have the same shape")

    midas_valid = np.isfinite(midas_value) & (midas_value > 0)
    depth_valid = np.isfinite(metric_depth_m)

    mask = midas_valid & depth_valid

    if not mask.any():
        raise ValueError("No valid pixels to fit the MiDaS scale")

    a = np.mean(metric_depth_m[mask] * midas_value[mask])
    return float(a)



def metric_depth_from_midas(relative_depth: np.ndarray,
                            scale_a: float) -> np.ndarray:
    """
    Generate a metric depth map from a MiDaS relative depth map using a single‑scale
    calibration factor ``a``:

        Z_metric = a / D_midas

    Pixels where the relative depth is <= 0 or not finite are set to ``np.nan``.

    Parameters
    ----------
    relative_depth : np.ndarray
        MiDaS relative depth predictions.
    scale_a : float
        Calibration factor obtained from :func:`fit_midas_single_scale`.

    Returns
    -------
    np.ndarray
        Metric depth map as a ``float32`` array.
    """
    out = np.full_like(relative_depth, np.nan, dtype=np.float32)
    valid = np.isfinite(relative_depth) & (relative_depth > 0)

    if valid.any():
        out[valid] = scale_a / relative_depth[valid]
    return out
