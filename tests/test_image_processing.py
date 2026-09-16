import numpy as np
import pytest

from src.processing.inpainting import inpaint_object


def test_inpaint_returns_same_shape_and_dtype():
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[9:11, 9:11] = True

    result = inpaint_object(image, mask)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_inpaint_does_not_modify_original_image():
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    original = image.copy()

    mask = np.zeros((20, 20), dtype=bool)
    mask[9:11, 9:11] = True

    inpaint_object(image, mask)

    assert np.array_equal(image, original)


def test_inpaint_accepts_integer_mask():
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[9:11, 9:11] = 1

    result = inpaint_object(image, mask)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_inpaint_without_mask_returns_equivalent_image():
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)

    result = inpaint_object(image, mask)

    assert np.array_equal(result, image)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((20, 20), dtype=np.uint8),
        np.zeros((20, 20, 4), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.float32),
    ],
)
def test_inpaint_rejects_invalid_images(image):
    mask = np.zeros((20, 20), dtype=bool)

    with pytest.raises((TypeError, ValueError)):
        inpaint_object(image, mask)


def test_inpaint_rejects_mask_shape_mismatch():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError):
        inpaint_object(image, mask)


def test_inpaint_rejects_non_2d_mask():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20, 1), dtype=bool)

    with pytest.raises(ValueError):
        inpaint_object(image, mask)


def test_inpaint_rejects_non_integer_non_boolean_mask():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.float32)

    with pytest.raises(TypeError):
        inpaint_object(image, mask)


@pytest.mark.parametrize("radius", [0, -1, 1.5, "3"])
def test_inpaint_rejects_invalid_radius(radius):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)

    with pytest.raises(ValueError):
        inpaint_object(image, mask, radius=radius)
