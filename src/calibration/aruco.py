"""Utility functions for detecting and estimating the pose of an ArUco marker.

The module exposes a single high‑level helper ``detect_and_estimate_pose`` that

* detects a specific ArUco marker ID in an RGB / BGR image, and
* estimates its 6‑DoF pose with respect to the camera using
  :func:`cv2.solvePnP`.

The function relies on the modern :class:`cv2.aruco.ArucoDetector` API
and is intentionally kept simple and beginner‑friendly.  It accepts a
camera intrinsic matrix ``K`` and distortion coefficients ``dist_coeffs``
and the real‑world side length of the marker in metres.  The return
values are the rotation vector ``rvec`` and translation vector ``tvec``
returned by :func:`cv2.solvePnP`, the Euclidean distance of the marker
from the camera (``np.linalg.norm(tvec)``), and the 2‑D image coordinates
of the marker corners.

Coordinate conventions
----------------------
* The camera coordinate system used by OpenCV is right‑handed with the
  X‑axis pointing to the right of the camera, the Y‑axis pointing
  downwards in the image, and the Z‑axis pointing forward (away
  from the camera).
* The marker coordinate system is defined so that the marker lies on
  the XY‑plane (Z = 0).  The origin is at the centre of the marker
  and the corners are ordered
  ``top‑left, top‑right, bottom‑right, bottom‑left``.

All distances returned by this module are in metres.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2

__all__ = ["detect_and_estimate_pose"]


def detect_and_estimate_pose(
    image: np.ndarray,
    marker_id: int,
    marker_length: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    dictionary_type: int = cv2.aruco.DICT_4X4_50,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Detect a single ArUco marker and estimate its pose.

    Parameters
    ----------
    image:
        RGB or BGR image as a ``numpy.ndarray`` of shape ``(H, W, 3)``.
    marker_id:
        The ArUco marker ID to search for.
    marker_length:
        Physical side length of the marker in metres.
    camera_matrix:
        Intrinsic camera matrix ``K`` of shape ``(3, 3)``.
    dist_coeffs:
        Distortion coefficients as returned by OpenCV camera calibration.
    dictionary_type:
        OpenCV predefined dictionary identifier (default
        ``cv2.aruco.DICT_4X4_50``).  The dictionary can be changed by the
        caller to support other marker sets.

    Returns
    -------
    rvec, tvec:
        Rotation and translation vectors returned by ``cv2.solvePnP``.
        ``rvec`` is a ``(3, 1)`` array of Rodrigues angles, ``tvec`` is a
        ``(3, 1)`` array representing the marker position in the camera
        coordinate system.
    distance:
        Euclidean distance from the camera centre to the marker centre
        (``np.linalg.norm(tvec)``) in metres.
    corners:
        The 2‑D image coordinates of the marker corners as a ``(4, 2)``
        ``numpy.ndarray`` in the order ``top‑left, top‑right,
        bottom‑right, bottom‑left``.

    Raises
    ------
    TypeError
        If the inputs are of unexpected type.
    ValueError
        If the inputs are invalid, the marker is not detected, or the
        requested ID is not present.
    """
    # ---------- Input validation ----------
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if not isinstance(marker_id, int) or marker_id < 0:
        raise ValueError("marker_id must be a non‑negative integer")

    if not isinstance(marker_length, (int, float, np.integer, np.floating)) or not np.isfinite(marker_length) or marker_length <= 0:
        raise ValueError("marker_length must be a positive, finite number")

    if not isinstance(camera_matrix, np.ndarray) or camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must be a numpy.ndarray of shape (3, 3)")
    if not np.issubdtype(camera_matrix.dtype, np.number) or not np.isfinite(camera_matrix).all():
        raise ValueError("camera_matrix must contain only numeric, finite values")

    if not isinstance(dist_coeffs, np.ndarray):
        raise TypeError("dist_coeffs must be a numpy.ndarray")
    # Accept 1‑D or 2‑D arrays; flatten to 1‑D
    dist_coeffs = np.asarray(dist_coeffs).reshape(-1)
    if dist_coeffs.size < 4:
        raise ValueError("dist_coeffs must contain at least 4 elements")
    if not np.issubdtype(dist_coeffs.dtype, np.number) or not np.isfinite(dist_coeffs).all():
        raise ValueError("dist_coeffs must contain only numeric, finite values")

    if not isinstance(dictionary_type, int):
        raise TypeError("dictionary_type must be an int constant from cv2.aruco")

    # ---------- Marker detection ----------
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) == 0:
        raise ValueError("no markers detected in the image")

    # Find the requested marker ID
    ids = ids.reshape(-1)
    idx = np.where(ids == marker_id)[0]
    if idx.size == 0:
        raise ValueError(f"marker ID {marker_id} not found in the image")
    idx = idx[0]

    # 2‑D image points of the marker corners
    img_corners = corners[idx].reshape(4, 2).astype(np.float32)

    # 3‑D world points of the marker corners (marker lies in XY‑plane, Z=0)
    half = marker_length / 2.0
    obj_corners = np.array(
        [
            [-half, -half, 0.0],  # top‑left
            [half, -half, 0.0],   # top‑right
            [half, half, 0.0],    # bottom‑right
            [-half, half, 0.0],   # bottom‑left
        ],
        dtype=np.float32,
    )

    # ---------- Pose estimation ----------
    success, rvec, tvec = cv2.solvePnP(
        obj_corners,
        img_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed to find a valid pose")

    distance = float(np.linalg.norm(tvec))

    return rvec, tvec, distance, img_corners

# End of file
