"""
Utility for classical OpenCV inpainting of object masks.

The public entry point is :func:`inpaint_object`, which accepts a RGB uint8
image and a boolean mask of pixels that should be inpainted.  The function
validates inputs, converts the mask to the 0/255 uint8 format required by
`cv2.inpaint`, and returns a **new** image without modifying the original
input.  The Telea algorithm (`cv2.INPAINT_TELEA`) is used with a
small configurable radius (default 3).
"""

import numpy as np
import cv2


def inpaint_object(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """
    Inpaint an RGB image using a binary mask.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3) and dtype ``np.uint8``.
    mask : np.ndarray
        Boolean mask of shape (H, W).  ``True`` indicates pixels that
        should be inpainted.  The mask can also be ``uint8`` with values 0/1.
    radius : int, optional
        Inpainting neighborhood radius in pixels.  Default is 3.

    Returns
    -------
    np.ndarray
        Inpainted RGB image of shape (H, W, 3) and dtype ``np.uint8``.
        The original image is left untouched.

    Raises
    ------
    TypeError
        If ``image`` is not a ``np.ndarray`` or not ``np.uint8``.
    ValueError
        If shapes are inconsistent, mask is not 2‑D, or ``radius`` is
        not a positive integer.
    """
    # --------------------------------------------------------------------
    # Validate image
    # --------------------------------------------------------------------
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy.ndarray, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"image dtype must be np.uint8, got {image.dtype}")

    # --------------------------------------------------------------------
    # Validate mask
    # --------------------------------------------------------------------
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy.ndarray, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2‑D, got {mask.ndim}‑D array")
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image height/width {image.shape[:2]}"
        )
    # Accept bool or integer types; convert to bool
    if mask.dtype != np.bool_:
        if not np.issubdtype(mask.dtype, np.integer):
            raise TypeError(
                f"mask dtype must be bool or an integer type, got {mask.dtype}"
            )
        mask_bool = mask.astype(bool)
    else:
        mask_bool = mask

    # Convert mask to 0/255 uint8 for OpenCV
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255

    # --------------------------------------------------------------------
    # Validate radius
    # --------------------------------------------------------------------
    if not isinstance(radius, int) or radius <= 0:
        raise ValueError(f"radius must be a positive integer, got {radius}")

    # --------------------------------------------------------------------
    # Inpaint
    # --------------------------------------------------------------------
    # cv2.inpaint returns a new array; the input image remains unchanged.
    result = cv2.inpaint(image, mask_uint8, radius, cv2.INPAINT_TELEA)

    return result
