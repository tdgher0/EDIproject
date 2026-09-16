import numpy as np
import pytest

from src.calibration.reference_object import (
    fit_midas_single_scale,
    metric_depth_from_midas,
    reference_depth_from_height,
)


def test_reference_depth_from_height():
    depth = reference_depth_from_height(
        reference_real_height_m=2.0,
        reference_pixel_height=100.0,
        fy=500.0,
    )
    assert depth == pytest.approx(10.0)


@pytest.mark.parametrize(
    "height,pixels,fy",
    [(0, 100, 500), (-1, 100, 500), (2, 0, 500), (2, -10, 500), (2, 100, 0)],
)
def test_reference_depth_from_height_rejects_invalid_inputs(
    height, pixels, fy
):
    with pytest.raises(ValueError):
        reference_depth_from_height(height, pixels, fy)


def test_fit_midas_single_scale():
    midas = np.array([2.0, 4.0, 5.0], dtype=np.float32)
    metric = np.array([3.0, 1.5, 1.2], dtype=np.float32)

    scale = fit_midas_single_scale(midas, metric)

    assert scale == pytest.approx(6.0)


def test_fit_midas_single_scale_ignores_invalid_midas_values():
    midas = np.array([2.0, 4.0, 0.0, np.nan], dtype=np.float32)
    metric = np.array([3.0, 1.5, 100.0, 100.0], dtype=np.float32)

    scale = fit_midas_single_scale(midas, metric)

    assert scale == pytest.approx(6.0)


def test_fit_midas_single_scale_rejects_shape_mismatch():
    midas = np.ones((2, 2), dtype=np.float32)
    metric = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        fit_midas_single_scale(midas, metric)


def test_fit_midas_single_scale_rejects_no_valid_pixels():
    midas = np.zeros((2, 2), dtype=np.float32)
    metric = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        fit_midas_single_scale(midas, metric)


def test_metric_depth_from_midas():
    relative = np.array([2.0, 4.0, 0.0, np.nan], dtype=np.float32)

    metric = metric_depth_from_midas(relative, 10.0)

    assert metric.dtype == np.float32
    assert metric[0] == pytest.approx(5.0)
    assert metric[1] == pytest.approx(2.5)
    assert np.isnan(metric[2])
    assert np.isnan(metric[3])
