import pytest

from src.segmentation.inference import _load_image_size


def test_load_image_size_from_project_config():
    assert _load_image_size("configs/model_config.yaml") == (512, 512)


def test_load_image_size_from_custom_config(tmp_path):
    config = tmp_path / "model_config.yaml"
    config.write_text(
        "train:\n"
        "  image_size: [256, 384]\n",
        encoding="utf-8",
    )

    assert _load_image_size(config) == (256, 384)


def test_load_image_size_missing_key(tmp_path):
    config = tmp_path / "model_config.yaml"
    config.write_text(
        "train:\n"
        "  batch_size: 4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        _load_image_size(config)


def test_load_image_size_missing_file(tmp_path):
    config = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        _load_image_size(config)
