#!/usr/bin/env python3
"""
Select a reproducible, stratified validation subset of 100 wall‑containing
training images from the SUN RGB‑D dataset.

The script performs the following steps:
1. Loads ``SUNRGBDMeta.mat`` to obtain the sensor type for each training
   index.
2. Determines whether a training image is wall‑containing by checking the
   ground‑truth mask image ``img-(5050+index).png`` for at least one pixel
   with value ``1``.
3. Stratifiy the selection to 37 ``kv2``, 21 ``kv1``, 32 ``xtion`` and
   10 ``realsense`` wall‑containing images using a fixed random seed
   (``42``) to guarantee reproducibility.
4. Writes the selected training indices to
   ``outputs/validation/validation_indices_100.txt`` – one index per line.
5. Prints the indices grouped by sensor and the final counts.

No model inference or modifications to existing project files are performed.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_metadata(mat_path: Path) -> np.ndarray:
    """Return the ``SUNRGBDMeta`` structured array from the given .mat file.

    The returned array has one entry per scene.  Indexing with an integer
    gives a ``numpy.void`` instance that behaves like a record with named
    fields.
    """
    mat = scipy.io.loadmat(str(mat_path))
    # The array is stored under the key 'SUNRGBDMeta'; the first element is a
    # 1‑D structured array.
    return mat["SUNRGBDMeta"][0]


def is_wall_containing(mask_path: Path) -> bool:
    """Return ``True`` if the ground‑truth mask contains at least one pixel
    with value ``1``.
    """
    mask = np.array(Image.open(mask_path))
    return np.any(mask == 1)

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    meta_path = project_root / "SUNRGBDtoolbox" / "SUNRGBDtoolbox" / "Metadata" / "SUNRGBDMeta.mat"
    meta = load_metadata(meta_path)

    # Sensor distribution requirements
    required_counts = {
        "kv2": 37,
        "kv1": 21,
        "xtion": 32,
        "realsense": 10,
    }

    # Map sensor type to list of wall‑containing training indices
    sensor_to_indices: dict[str, list[int]] = {sensor: [] for sensor in required_counts}

    gt_dir = project_root / "data" / "raw" / "sunrgbd_labels"

    print("Scanning training images for wall‑containing masks…")
    # Training indices range from 1 to 5285 (inclusive)
    for idx in range(1, 5286):
        # Sensor type is stored in meta[5050 + idx - 1]
        record = meta[5050 + idx - 1]
        st = record["sensorType"][0]
        # Convert to a plain string
        if isinstance(st, (bytes, bytearray)):
            sensor_type = st.decode("utf-8")
        else:
            sensor_type = str(st)
        sensor_type = sensor_type.lower()

        if sensor_type not in sensor_to_indices:
            # Skip sensors that are not part of the stratification
            continue

        gt_index = 5050 + idx
        gt_path = gt_dir / f"img-{gt_index:06d}.png"
        if not gt_path.exists():
            continue
        if is_wall_containing(gt_path):
            sensor_to_indices[sensor_type].append(idx)

    # Validate availability
    for sensor, needed in required_counts.items():
        available = len(sensor_to_indices[sensor])
        if available < needed:
            raise RuntimeError(
                f"Not enough wall‑containing images for sensor '{sensor}': required {needed}, available {available}"
            )

    # Reproducible random selection
    random.seed(42)
    selected: list[int] = []
    selected_by_sensor: dict[str, list[int]] = {s: [] for s in required_counts}
    for sensor, needed in required_counts.items():
        choices = random.sample(sensor_to_indices[sensor], needed)
        selected.extend(choices)
        selected_by_sensor[sensor] = sorted(choices)

    if len(set(selected)) != 100:
        raise RuntimeError("Selected indices are not unique or count is not 100")

    # Write the selected indices to file
    out_dir = project_root / "outputs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_indices_100.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for idx in sorted(selected):
            f.write(f"{idx}\n")

    # Output summary
    print("\nSelected indices by sensor:")
    for sensor in sorted(required_counts):
        print(f"{sensor}: {selected_by_sensor[sensor]}")

    print(f"\nTotal selected indices: {len(selected)}")
    print(f"Indices written to: {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
