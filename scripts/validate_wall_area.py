"""Validate wall‑area estimation on selected SUN RGB‑D training images.

This script demonstrates how the trained segmentation model and the depth‑based
area calculator work together.  It processes a handful of images, compares the
predicted 3‑D wall area against the ground truth mask, and prints a summary
table together with a CSV file containing the results.

The implementation follows the specification in the user request and is
intentionally beginner‑friendly – all heavy lifting is delegated to the
:mod:`src.segmentation.inference` and :mod:`src.measurement.wall_area`
modules.

Usage
-----
```
python scripts/validate_wall_area.py
```

The script can also be called with a custom list of training indices::

    python scripts/validate_wall_area.py 1 903 2000

If no indices are supplied the default list ``[1, 903, 2000, 3000, 4000, 5285]``
is used.
"""

from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import scipy.io
from PIL import Image

# Local imports – the modules are part of the same project.
# Importing from src will work because the repository root is on sys.path.
from src.segmentation.inference import predict_wall_mask
from src.measurement.wall_area import calculate_wall_area_from_depth

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_metadata(mat_path: str | Path) -> np.ndarray:
    """Return the structured array ``SUNRGBDMeta`` from the .mat file.

    The returned array has one entry per scene.  Indexing with an integer gives a
    ``numpy.void`` instance that behaves like a record with named fields.
    """
    mat = scipy.io.loadmat(str(mat_path))
    # The array is stored under the key 'SUNRGBDMeta'; the first element is a
    # 1‑D structured array.
    meta = mat["SUNRGBDMeta"][0]
    return meta


def convert_depth(raw_depth: Image.Image) -> np.ndarray:
    """Convert the 16‑bit PNG from the SUNRGBD toolbox to a metric depth map.

    The official conversion is::

        depth = (depth >> 3) | (depth << 13)
        depth = depth.astype(float32) / 1000
        depth[depth > 8] = 8

    The result is a ``float32`` array in metres.
    """
    # ``Image`` will return a 16‑bit array; convert to ``uint16`` just in case.
    depth_uint16 = np.array(raw_depth, dtype=np.uint16)
    depth = (depth_uint16 >> 3) | (depth_uint16 << 13)
    depth = depth.astype(np.float32) / 1000.0
    depth[depth > 8] = 8.0
    return depth


def process_one_index(
    index: int,
    meta: np.ndarray,
    depth_dir: Path,
    rgb_dir: Path,
    gt_dir: Path,
    checkpoint_path: Path,
) -> dict:
    """Process a single training index and return the metrics.

    Parameters
    ----------
    index:
        The training image number (1‑based) as used by the user.
    meta:
        The metadata structured array.
    depth_dir, rgb_dir, gt_dir:
        Directories containing the depth, RGB, and ground‑truth mask files.
    checkpoint_path:
        Path to the model checkpoint.

    Returns
    -------
    dict
        Dictionary with keys:
        ``index``, ``sensor_type``, ``width``, ``height``, ``pred_area``, ``gt_area``, ``abs_error``, ``pct_error``.
    """
    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    rgb_path = rgb_dir / f"img-{index:06d}.jpg"
    depth_path = depth_dir / f"{index}.png"
    # Ground‑truth mask is stored with index+5050.
    gt_index = 5050 + index
    gt_path = gt_dir / f"img-{gt_index:06d}.png"

    # -----------------------------------------------------------------------
    # Load image metadata
    # -----------------------------------------------------------------------
    record_idx = 5050 + index - 1  # zero‑based
    record = meta[record_idx]
    sensor_type = record["sensorType"][0]
    K = record["K"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # -----------------------------------------------------------------------
    # Image dimensions
    # -----------------------------------------------------------------------
    with Image.open(rgb_path) as img:
        orig_w, orig_h = img.size

    # -----------------------------------------------------------------------
    # Depth map
    # -----------------------------------------------------------------------
    with Image.open(depth_path) as depth_img:
        depth = convert_depth(depth_img)

    # -----------------------------------------------------------------------
    # Predicted wall mask
    # -----------------------------------------------------------------------
    pred_mask = predict_wall_mask(str(rgb_path), str(checkpoint_path))

    # -----------------------------------------------------------------------
    # Ground‑truth mask
    # -----------------------------------------------------------------------
    with Image.open(gt_path) as gt_img:
        gt_array = np.array(gt_img)
    gt_mask = (gt_array == 1)

    # -----------------------------------------------------------------------
    # Wall‑area calculations
    # -----------------------------------------------------------------------
    gt_contains_wall = np.any(gt_mask)
    pred_contains_wall = np.any(pred_mask)
    if not gt_contains_wall:
        # No ground‑truth wall pixels.
        pred_area = None
        gt_area = None
        abs_error = None
        pct_error = None
    else:
        pred_area = calculate_wall_area_from_depth(pred_mask, depth, fx, fy, cx, cy, max_depth_jump=0.025)
        gt_area = calculate_wall_area_from_depth(gt_mask, depth, fx, fy, cx, cy, max_depth_jump=0.025)
        abs_error = abs(pred_area - gt_area)
        pct_error = 0.0 if gt_area == 0 else abs_error / gt_area * 100.0

    return {
        "index": index,
        "sensor_type": sensor_type,
        "width": orig_w,
        "height": orig_h,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "abs_error": abs_error,
        "pct_error": pct_error,
        "gt_contains_wall": gt_contains_wall,
        "pred_contains_wall": pred_contains_wall,
    }

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate wall‑area measurement on SUN RGB‑D images.")
    parser.add_argument(
        "indices",
        nargs="*",
        type=int,
        default=[1, 903, 2000, 3000, 4000, 5285],
        help="Training indices to process (default: %(default)s)",
    )
    args = parser.parse_args()

    # Paths – all are relative to the repository root.
    project_root = Path(__file__).resolve().parent.parent
    depth_dir = project_root / "data/raw/sunrgbd_train_depth"
    rgb_dir = project_root / "data/raw/sunrgbd"
    gt_dir = project_root / "data/raw/sunrgbd_labels"
    mat_path = project_root / "SUNRGBDtoolbox/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat"
    checkpoint_path = project_root / "models/checkpoints/best_model.pt"

    meta = load_metadata(mat_path)

    results: list[dict] = []
    for idx in args.indices:
        try:
            res = process_one_index(
                index=idx,
                meta=meta,
                depth_dir=depth_dir,
                rgb_dir=rgb_dir,
                gt_dir=gt_dir,
                checkpoint_path=checkpoint_path,
            )
            results.append(res)
        except Exception as e:  # pragma: no cover – any unexpected error
            print(f"Error processing index {idx}: {e}")

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    header = [
        "Index",
        "Sensor",
        "W",
        "H",
        "Pred Area (m2)",
        "GT Area (m2)",
        "Abs Error (m2)",
        "Pct Error (%)",
    ]
    print("\nValidation results:\n")
    print("{:<6} {:<10} {:>5} {:>5} {:>16} {:>16} {:>18} {:>14}".format(*header))
    for r in results:
        pred_area_str = f"{r['pred_area']:.4f}" if r['pred_area'] is not None else "-"
        gt_area_str = f"{r['gt_area']:.4f}" if r['gt_area'] is not None else "-"
        abs_error_str = f"{r['abs_error']:.4f}" if r['abs_error'] is not None else "-"
        pct_error_str = f"{r['pct_error']:.2f}" if r['pct_error'] is not None else "-"
        print(
            "{:<6} {:<10} {:>5} {:>5} {:>16} {:>16} {:>18} {:>14}".format(
                r["index"],
                r["sensor_type"],
                r["width"],
                r["height"],
                pred_area_str,
                gt_area_str,
                abs_error_str,
                pct_error_str,
            )
        )

    # Count no-wall images
    no_wall_count = sum(1 for r in results if not r["gt_contains_wall"])
    no_wall_correct = sum(1 for r in results if not r["gt_contains_wall"] and not r["pred_contains_wall"])

    # Separate area metrics for wall images
    wall_results = [r for r in results if r["gt_contains_wall"]]
    if wall_results:
        wall_abs_mean = np.mean([r["abs_error"] for r in wall_results])
        wall_pct_mean = np.mean([r["pct_error"] for r in wall_results])
    else:
        wall_abs_mean = 0.0
        wall_pct_mean = 0.0

    print(f"\nNo-wall images: total {no_wall_count}")
    print(f"Correctly predicted empty: {no_wall_correct}")
    print("\nWall-containing images: mean absolute error {:.4f} m2".format(wall_abs_mean))
    print("Mean percentage error {:.2f}%".format(wall_pct_mean))

    # Write CSV
    output_dir = project_root / "outputs/validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "wall_area_validation.csv"

    if results:
        # Ensure the CSV includes the new fields
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults written to {csv_path}")
    else:
        print("\nNo successful validation results to write.")


if __name__ == "__main__":  # pragma: no cover
    main()
