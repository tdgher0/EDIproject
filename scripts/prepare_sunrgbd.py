from pathlib import Path
from PIL import Image
import numpy as np
import argparse


TRAIN_IMAGE_DIR = Path("data/raw/sunrgbd")
TEST_IMAGE_DIR = Path("data/raw/sunrgbd_test")
LABEL_DIR = Path("data/raw/sunrgbd_labels")

OUTPUT_DIR = Path("data/processed/sunrgbd")


def create_wall_mask(label_image):
    """Create a binary wall mask from a SUN RGB-D 37-class label."""
    label = np.array(label_image)

    # SUN RGB-D 37-class label:
    # Class 1 = wall
    wall_mask = np.where(label == 1, 255, 0).astype(np.uint8)

    return wall_mask


def process_sample(image_number, split):
    """Process one SUN RGB-D image and its matching label."""

    if split == "train":
        image_dir = TRAIN_IMAGE_DIR

        # Training image 000001 -> label 005051
        label_number = image_number + 5050

    else:
        image_dir = TEST_IMAGE_DIR

        # Test image 000001 -> label 000001
        label_number = image_number

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

    wall_mask = create_wall_mask(label)

    output_image_dir = OUTPUT_DIR / split / "images"
    output_mask_dir = OUTPUT_DIR / split / "masks"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    output_image_path = output_image_dir / image_name
    output_mask_path = output_mask_dir / f"img-{image_number:06d}.png"

    image.save(output_image_path)
    Image.fromarray(wall_mask).save(output_mask_path)

    return True


def main(split, num_samples):

    total_samples = 5285 if split == "train" else 5050

    print(f"Preparing SUN RGB-D {split} data...")

    processed = 0

    for image_number in range(1, total_samples + 1):

        if num_samples > 0 and processed >= num_samples:
            break

        output_image = (
            OUTPUT_DIR / split / "images" /
            f"img-{image_number:06d}.jpg"
        )

        output_mask = (
            OUTPUT_DIR / split / "masks" /
            f"img-{image_number:06d}.png"
        )

        if output_image.exists() and output_mask.exists():
            print(f"Skipping existing sample {image_number}")
            processed += 1
            continue

        success = process_sample(image_number, split)

        if success:
            processed += 1
            print(f"Processed {split} sample {processed}")

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Images: {OUTPUT_DIR / split / 'images'}")
    print(f"Masks: {OUTPUT_DIR / split / 'masks'}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Prepare SUN RGB-D wall segmentation data."
    )

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        required=True,
        help="Dataset split to process.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples. Use 0 for the entire split.",
    )

    args = parser.parse_args()

    main(args.split, args.samples)