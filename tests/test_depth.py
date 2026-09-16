import numpy as np

from src.depth.midas import save_depth_visualization, save_raw_depth


def test_save_depth_visualization(tmp_path):
    depth = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    output_path = tmp_path / "depth.png"

    save_depth_visualization(depth, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_raw_depth(tmp_path):
    depth = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    output_path = tmp_path / "depth.npy"

    save_raw_depth(depth, output_path)

    assert output_path.exists()

    loaded = np.load(output_path)

    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, depth)
