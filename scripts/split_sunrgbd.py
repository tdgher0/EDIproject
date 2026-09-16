from pathlib import Path
import random
import shutil


SOURCE_DIR = Path("data/processed/sunrgbd/train")

TRAIN_DIR = Path("data/processed/sunrgbd/train_split")
VAL_DIR = Path("data/processed/sunrgbd/val")


TRAIN_RATIO = 0.90
SEED = 42


def main():
    source_images = SOURCE_DIR / "images"
    source_masks = SOURCE_DIR / "masks"

    train_images = TRAIN_DIR / "images"
    train_masks = TRAIN_DIR / "masks"

    val_images = VAL_DIR / "images"
    val_masks = VAL_DIR / "masks"

    train_images.mkdir(parents=True, exist_ok=True)
    train_masks.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)
    val_masks.mkdir(parents=True, exist_ok=True)

    images = sorted(source_images.glob("*.jpg"))

    if not images:
        print("No training images found.")
        return

    random.seed(SEED)
    random.shuffle(images)

    train_count = int(len(images) * TRAIN_RATIO)

    train_files = images[:train_count]
    val_files = images[train_count:]

    print(f"Total images: {len(images)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    print("\nCreating training split...")

    for image_path in train_files:
        mask_path = source_masks / f"{image_path.stem}.png"

        shutil.copy2(image_path, train_images / image_path.name)
        shutil.copy2(mask_path, train_masks / mask_path.name)

    print("Training split complete.")

    print("\nCreating validation split...")

    for image_path in val_files:
        mask_path = source_masks / f"{image_path.stem}.png"

        shutil.copy2(image_path, val_images / image_path.name)
        shutil.copy2(mask_path, val_masks / mask_path.name)

    print("Validation split complete.")

    print("\nDone.")
    print(f"Training:   {TRAIN_DIR}")
    print(f"Validation: {VAL_DIR}")


if __name__ == "__main__":
    main()