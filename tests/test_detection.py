import numpy as np
import pytest

from src.detection.object_mask import (
    get_object_wall_overlap,
    remove_objects_from_wall,
)


def test_get_object_wall_overlap():
    wall = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    object_mask = np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    result = get_object_wall_overlap(
        wall,
        [{"mask": object_mask}],
    )

    expected = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    assert np.array_equal(result, expected)


def test_multiple_object_masks_are_combined():
    wall = np.ones((3, 3), dtype=bool)

    object_1 = np.zeros((3, 3), dtype=bool)
    object_1[0, 0] = True

    object_2 = np.zeros((3, 3), dtype=bool)
    object_2[2, 2] = True

    result = get_object_wall_overlap(
        wall,
        [
            {"mask": object_1},
            {"mask": object_2},
        ],
    )

    expected = np.zeros((3, 3), dtype=bool)
    expected[0, 0] = True
    expected[2, 2] = True

    assert np.array_equal(result, expected)


def test_objects_outside_wall_are_ignored():
    wall = np.zeros((3, 3), dtype=bool)
    wall[1, 1] = True

    object_mask = np.ones((3, 3), dtype=bool)

    result = get_object_wall_overlap(
        wall,
        [{"mask": object_mask}],
    )

    expected = np.zeros((3, 3), dtype=bool)
    expected[1, 1] = True

    assert np.array_equal(result, expected)


def test_empty_detections_return_empty_mask():
    wall = np.ones((3, 3), dtype=bool)

    result = get_object_wall_overlap(wall, [])

    assert result.dtype == bool
    assert not np.any(result)


def test_missing_detection_mask_raises_key_error():
    wall = np.ones((3, 3), dtype=bool)

    with pytest.raises(KeyError):
        get_object_wall_overlap(
            wall,
            [{"class_id": 0}],
        )


def test_detection_shape_mismatch_raises_value_error():
    wall = np.ones((3, 3), dtype=bool)
    object_mask = np.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError):
        get_object_wall_overlap(
            wall,
            [{"mask": object_mask}],
        )


def test_wall_mask_must_be_2d():
    wall = np.ones((3, 3, 1), dtype=bool)

    with pytest.raises(ValueError):
        get_object_wall_overlap(wall, [])


def test_remove_objects_from_wall():
    wall = np.ones((3, 3), dtype=bool)

    object_mask = np.zeros((3, 3), dtype=bool)
    object_mask[1, 1] = True
    object_mask[2, 2] = True

    result = remove_objects_from_wall(
        wall,
        [{"mask": object_mask}],
    )

    expected = np.ones((3, 3), dtype=bool)
    expected[1, 1] = False
    expected[2, 2] = False

    assert np.array_equal(result, expected)


def test_remove_objects_does_not_remove_non_wall_pixels():
    wall = np.zeros((3, 3), dtype=bool)
    wall[0, 0] = True

    object_mask = np.ones((3, 3), dtype=bool)

    result = remove_objects_from_wall(
        wall,
        [{"mask": object_mask}],
    )

    expected = np.zeros((3, 3), dtype=bool)
    expected[0, 0] = False

    assert np.array_equal(result, expected)
