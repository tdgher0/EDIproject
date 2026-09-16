import numpy as np
import pytest

from src.pipeline.pipeline import run_pipeline


def make_placeholder_image(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    return image


def test_pipeline_rejects_missing_image():
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            "does_not_exist.jpg",
            1059,
            529.5,
            529.5,
            365,
            265,
        )


@pytest.mark.parametrize(
    "parameter_index",
    [1, 2, 3, 4, 5],
)
def test_pipeline_rejects_non_numeric_inputs(parameter_index, tmp_path):
    image = make_placeholder_image(tmp_path)

    values = [
        image,
        1059,
        529.5,
        529.5,
        365,
        265,
    ]

    values[parameter_index] = "invalid"

    with pytest.raises(TypeError):
        run_pipeline(*values)


@pytest.mark.parametrize(
    "parameter_index",
    [1, 2, 3, 4, 5],
)
def test_pipeline_rejects_non_finite_inputs(parameter_index, tmp_path):
    image = make_placeholder_image(tmp_path)

    values = [
        image,
        1059,
        529.5,
        529.5,
        365,
        265,
    ]

    values[parameter_index] = np.nan

    with pytest.raises(ValueError):
        run_pipeline(*values)


def test_pipeline_rejects_zero_calibration_factor(tmp_path):
    image = make_placeholder_image(tmp_path)

    with pytest.raises(ValueError):
        run_pipeline(
            image,
            0,
            529.5,
            529.5,
            365,
            265,
        )


def test_pipeline_rejects_negative_calibration_factor(tmp_path):
    image = make_placeholder_image(tmp_path)

    with pytest.raises(ValueError):
        run_pipeline(
            image,
            -1,
            529.5,
            529.5,
            365,
            265,
        )


@pytest.mark.parametrize(
    "fx,fy",
    [
        (0, 529.5),
        (-1, 529.5),
        (529.5, 0),
        (529.5, -1),
    ],
)
def test_pipeline_rejects_non_positive_focal_lengths(tmp_path, fx, fy):
    image = make_placeholder_image(tmp_path)

    with pytest.raises(ValueError):
        run_pipeline(
            image,
            1059,
            fx,
            fy,
            365,
            265,
        )


def test_pipeline_rejects_negative_depth_jump(tmp_path):
    image = make_placeholder_image(tmp_path)

    with pytest.raises(ValueError):
        run_pipeline(
            image,
            1059,
            529.5,
            529.5,
            365,
            265,
            max_depth_jump=-0.1,
        )
