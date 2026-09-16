from datasets import load_dataset
from pathlib import Path
from PIL import Image
import numpy as np
import argparse


OUTPUT_DIR = Path("data/processed/nyuv2")


def create_wall_mask(semantic_image):
    """Convert a NYUv2 semantic mask into a binary wall mask."""
    semantic = np.array(semantic_image)

    # NYUv2 stores class labels as 16-bit normalized values.
    class_ids = np.rint((semantic / 65535.0) * 40).astype(np.uint8)

    # Class ID 1 = wall.
    wall_mask = np.where(class_ids == 1, 255, 0).astype(np.uint8)

    return wall_mask


def main(split, num_samples):
    split_dir = OUTPUT_DIR / split
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading NYUv2 {split} split...")

    dataset = load_dataset(
        "jagennath-hari/nyuv2",
        split=split,
        streaming=True,
    )

    if num_samples == 0:
        print("Processing all samples...")
    else:
        print(f"Processing {num_samples} samples...")

    saved = 0

    for index, sample in enumerate(dataset):

        if num_samples > 0 and saved >= num_samples:
            break

        image_path = image_dir / f"{index:05d}.png"
        mask_path = mask_dir / f"{index:05d}.png"

        # Skip samples that were already processed.
        if image_path.exists() and mask_path.exists():
            print(f"Skipping existing sample {index + 1}")
            saved += 1
            continue

        image = sample["rgb"]
        semantic = sample["semantic"]

        wall_mask = create_wall_mask(semantic)

        image.save(image_path)
        Image.fromarray(wall_mask).save(mask_path)

        saved += 1

        print(f"Saved {split} sample {saved}")

    print("\nDone.")
    print(f"Processed: {saved}")
    print(f"Images: {image_dir}")
    print(f"Masks:  {mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare NYUv2 RGB images and wall masks."
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
        help="NYUv2 dataset split to process.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to process. Use 0 for the entire split.",
    )

    args = parser.parse_args()

    main(args.split, args.samples)