"""Prepare SUN RGB‑D opening‑segmentation dataset.

The script reads the deterministic split from ``data/processed/sunrgbd`` and
copies the corresponding RGB images from the raw directories.
It generates an opening mask (door = 1, window = 2, background = 0) from the
semantic labels stored in ``data/raw/sunrgbd_labels``.

The mapping rules are:

* **train_split** and **val** – source image ``img-<N>.jpg`` maps to
  label ``img-<N+5050>.png``.
* **test** – source image ``img-<N>.jpg`` maps to label ``img-<N>.png``.

The resulting dataset is written to
``data/processed/sunrgbd_openings/<split>/images`` and
``data/processed/sunrgbd_openings/<split>/masks``.

Usage example:
```
python scripts/prepare_opening_dataset.py --split train_split --samples 0
```

All masks are uint8 PNGs containing only values 0, 1 and 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

# --- Configuration -----------------------------------------------------------
ROOT_DIR = Path("data")

RAW_IMAGE_DIR = ROOT_DIR / "raw" / "sunrgbd"
RAW_TEST_IMAGE_DIR = ROOT_DIR / "raw" / "sunrgbd_test"
LABEL_DIR = ROOT_DIR / "raw" / "sunrgbd_labels"

OUTPUT_ROOT = ROOT_DIR / "processed" / "sunrgbd_openings"

# Mapping offsets for training/validation images
TRAIN_LABEL_OFFSET = 5050

# Class IDs for door and window in SUN RGB‑D
CLASS_Door = 8
CLASS_Window = 9

# ---------------------------------------------------------------------------

def create_opening_mask(label_image: Image.Image) -> np.ndarray:
    """Create a 0/1/2 mask from a SUN RGB‑D semantic label.

    Parameters
    ----------
    label_image : PIL.Image.Image
        The semantic label image.

    Returns
    -------
    np.ndarray
        A uint8 array where 0 = background, 1 = door, 2 = window.
    """
    label_arr = np.array(label_image)
    mask = np.zeros_like(label_arr, dtype=np.uint8)
    mask[label_arr == CLASS_Door] = 1
    mask[label_arr == CLASS_Window] = 2
    return mask


def process_sample(
    image_number: int,
    split: str,
    output_image_dir: Path,
    output_mask_dir: Path,
) -> bool:
    """Process one sample for the opening dataset.

    Parameters
    ----------
    image_number : int
        The numeric index of the image (zero‑padded 6‑digit when written).
    split : str
        One of ``train_split``, ``val`` or ``test``.
    output_image_dir : Path
        Directory to write the RGB image.
    output_mask_dir : Path
        Directory to write the opening mask.

    Returns
    -------
    bool
        ``True`` if the sample was processed successfully.
    """
    # Determine source directories based on split
    if split in {"train_split", "val"}:
        image_dir = RAW_IMAGE_DIR
        label_number = image_number + TRAIN_LABEL_OFFSET
    elif split == "test":
        image_dir = RAW_TEST_IMAGE_DIR
        label_number = image_number
    else:
        raise ValueError(f"Unknown split {split}")

    image_name = f"img-{image_number:06d}.jpg"
    label_name = f"img-{label_number:06d}.png"

    image_path = image_dir / image_name
    label_path = LABEL_DIR / label_name

    if not image_path.exists():
        print(f"Missing image: {image_name}")
        return False
    if not label_path.exists():
        print(f"Missing label: {label_name}")
        return False

    image = Image.open(image_path).convert("RGB")
    label = Image.open(label_path)

    # Validate that the RGB image and label share the same spatial dimensions
    image_size = (image.height, image.width)
    label_size = (label.height, label.width)
    if image_size != label_size:
        raise ValueError(
            f"Image dimensions {image_size} do not match label dimensions {label_size} for image {image_number:06d}"
        )

    mask = create_opening_mask(label)

    # Validation checks
    if not np.all(np.isin(mask, [0, 1, 2])):
        print(f"Warning: mask contains unexpected values for image {image_number:06d}")
    if mask.shape != np.array(label).shape:
        print(f"Warning: mask shape {mask.shape} != label shape {np.array(label).shape} for image {image_number:06d}")

    # Write output
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    output_image_path = output_image_dir / image_name
    output_mask_path = output_mask_dir / f"img-{image_number:06d}.png"

    image.save(output_image_path)
    Image.fromarray(mask).save(output_mask_path)

    return True


def load_image_numbers(split_dir: Path) -> Iterable[int]:
    """Yield image numbers (int) from ``split_dir/images``.

    The function assumes the filenames follow ``img-<NNNNNN>.jpg``.
    """
    for p in sorted(split_dir.glob("img-*.jpg")):
        try:
            number = int(p.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        yield number


def main(split: str, samples: int) -> None:
    split_processed_dir = ROOT_DIR / "processed" / "sunrgbd" / split
    if not split_processed_dir.exists():
        raise FileNotFoundError(f"Split directory {split_processed_dir} does not exist")

    image_numbers = list(load_image_numbers(split_processed_dir / "images"))
    total_available = len(image_numbers)

    print(f"Preparing SUN RGB‑D opening dataset – split: {split}")
    print(f"Found {total_available} images in the original split")

    output_image_dir = OUTPUT_ROOT / split / "images"
    output_mask_dir = OUTPUT_ROOT / split / "masks"

    processed = 0
    for idx, num in enumerate(image_numbers, 1):
        if samples > 0 and processed >= samples:
            break

        # Skip if both image and mask already exist
        if (output_image_dir / f"img-{num:06d}.jpg").exists() and \
           (output_mask_dir / f"img-{num:06d}.png").exists():
            print(f"Skipping existing sample {num:06d}")
            processed += 1
            continue

        success = process_sample(num, split, output_image_dir, output_mask_dir)
        if success:
            processed += 1
            print(f"Processed {split} sample {processed}/{min(samples or total_available, total_available)}")

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Images: {output_image_dir}")
    print(f"Masks: {output_mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SUN RGB‑D opening‑segmentation data.")
    parser.add_argument("--split", choices=["train_split", "val", "test"], required=True, help="Dataset split to process.")
    parser.add_argument("--samples", type=int, default=0, help="Number of samples. Use 0 for all.")
    args = parser.parse_args()
    main(args.split, args.samples)
