import numpy as np
import pytest

from src.measurement.wall_area import (
    calculate_baseline_wall_area,
    calculate_wall_area_from_depth,
    calculate_wall_pixel_area,
)


def test_calculate_wall_pixel_area():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert calculate_wall_pixel_area(mask) == 2


def test_calculate_wall_pixel_area_requires_2d():
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_wall_pixel_area(mask)


def test_calculate_wall_pixel_area_rejects_empty():
    mask = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_wall_pixel_area(mask)


def test_calculate_baseline_wall_area():
    mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    assert calculate_baseline_wall_area(mask, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("scale", [0, -0.1])
def test_calculate_baseline_wall_area_rejects_non_positive_scale(scale):
    mask = np.ones((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_baseline_wall_area(mask, scale)


def test_calculate_wall_area_from_depth_flat_wall():
    mask = np.ones((2, 2), dtype=np.uint8)
    depth = np.full((2, 2), 2.0, dtype=np.float32)

    area = calculate_wall_area_from_depth(
        mask,
        depth,
        fx=2.0,
        fy=2.0,
        cx=0.5,
        cy=0.5,
    )

    expected_area = 1.0
    assert np.isfinite(area)
    assert area == pytest.approx(expected_area)


def test_calculate_wall_area_from_depth_rejects_shape_mismatch():
    mask = np.ones((2, 2), dtype=np.uint8)
    depth = np.ones((3, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        calculate_wall_area_from_depth(
            mask, depth, 2.0, 2.0, 0.5, 0.5
        )


def test_calculate_wall_area_from_depth_rejects_non_positive_depth():
    mask = np.ones((2, 2), dtype=np.uint8)
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    depth[0, 0] = 0.0

    with pytest.raises(ValueError):
        calculate_wall_area_from_depth(
            mask, depth, 2.0, 2.0, 0.5, 0.5
        )


def test_calculate_wall_area_from_depth_rejects_empty_mask():
    mask = np.zeros((2, 2), dtype=np.uint8)
    depth = np.full((2, 2), 2.0, dtype=np.float32)

    with pytest.raises(ValueError):
        calculate_wall_area_from_depth(
            mask, depth, 2.0, 2.0, 0.5, 0.5
        )


def test_calculate_wall_area_from_depth_rejects_negative_depth_jump():
    mask = np.ones((2, 2), dtype=np.uint8)
    depth = np.full((2, 2), 2.0, dtype=np.float32)

    with pytest.raises(ValueError):
        calculate_wall_area_from_depth(
            mask,
            depth,
            2.0,
            2.0,
            0.5,
            0.5,
            max_depth_jump=-0.01,
        )
