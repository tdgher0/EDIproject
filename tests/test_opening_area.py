import numpy as np
import pytest

from src.measurement.opening_area import calculate_opening_area_from_depth


def test_flat_opening_area():
    mask = np.ones((2, 2), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)

    area = calculate_opening_area_from_depth(
        mask,
        depth,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        max_depth_jump=None,
    )

    assert area == pytest.approx(1.0)


def test_opening_area_rejects_shape_mismatch():
    mask = np.ones((2, 2), dtype=np.uint8)
    depth = np.ones((3, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        calculate_opening_area_from_depth(
            mask, depth, 1.0, 1.0, 0.0, 0.0
        )


def test_opening_area_rejects_empty_mask():
    mask = np.zeros((2, 2), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        calculate_opening_area_from_depth(
            mask, depth, 1.0, 1.0, 0.0, 0.0
        )


@pytest.mark.parametrize(
    "fx,fy",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
    ],
)
def test_opening_area_rejects_invalid_focal_lengths(fx, fy):
    mask = np.ones((2, 2), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        calculate_opening_area_from_depth(
            mask, depth, fx, fy, 0.0, 0.0
        )


def test_opening_area_rejects_non_positive_depth():
    mask = np.ones((2, 2), dtype=np.uint8)
    depth = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        calculate_opening_area_from_depth(
            mask, depth, 1.0, 1.0, 0.0, 0.0
        )


def test_depth_jump_filter_excludes_discontinuous_triangle():
    mask = np.ones((2, 2), dtype=np.uint8)
    depth = np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )

    filtered_area = calculate_opening_area_from_depth(
        mask,
        depth,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        max_depth_jump=0.1,
    )

    unfiltered_area = calculate_opening_area_from_depth(
        mask,
        depth,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        max_depth_jump=None,
    )

    assert filtered_area > 0
    assert unfiltered_area > filtered_area
