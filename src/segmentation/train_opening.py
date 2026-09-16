"""Dedicated training script for the 3‑class SUN RGB‑D opening segmentation model.

The script trains a UNetResNet34 (in the main project code) on the opening‑segmentation
dataset produced by :mod:`scripts.prepare_opening_dataset`.

Training configuration is local to this file and does **not** modify
``configs/model_config.yaml``.

Example usage::

    python src/segmentation/train_opening.py

It will write the best model to
``models/checkpoints/opening_augmented_best_model.pt``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
import torchvision.transforms.functional as TF

def train_transform(image: torch.Tensor, mask: torch.Tensor):
    """
    Random geometric augmentation applied identically to image and mask.

    - 50% chance of horizontal flip
    - Random rotation between -10° and +10°
      * Image: bilinear interpolation
      * Mask: nearest-neighbor (preserves integer class indices)

    Returns the transformed (image, mask) pair.  Mask is cast back to torch.long.
    """
    if torch.rand(1).item() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    angle = torch.empty(1).uniform_(-10, 10).item()
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
    # Rotate the mask with an added channel dimension,
    # then remove that dimension again.
    mask = TF.rotate(mask.unsqueeze(0),
                     angle,
                     interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
    return image, mask.long()
from torch.utils.data import DataLoader

from src.segmentation.opening_dataset import OpeningSegmentationDataset
from src.segmentation.model import UNetResNet34

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_SIZE = (512, 512)  # (height, width)
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

DATA_ROOT = Path("data") / "processed" / "sunrgbd_openings"
TRAIN_DIR = DATA_ROOT / "train_split"
VAL_DIR = DATA_ROOT / "val"

CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / "opening_augmented_best_model.pt"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Custom collate that resizes each image/mask pair to ``IMAGE_SIZE`` before stacking.

    The original dataset returns images of varying resolutions.  The training loop
    expects all tensors to be of the same spatial size.  We therefore resize each
    image with bilinear interpolation and each mask with nearest‑neighbor
    interpolation (preserving integer class indices) inside the collate
    function.

    Parameters
    ----------
    batch : list[tuple[torch.Tensor, torch.Tensor]]
        List of ``(image, mask)`` tuples.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``images`` of shape ``[B, 3, 512, 512]`` and ``masks`` of shape
        ``[B, 512, 512]``.
    """
    images, masks = zip(*batch)

    resized_images = []
    resized_masks = []
    for img, mask in zip(images, masks):
        # Ensure tensors are contiguous.
        img = img.contiguous()
        mask = mask.contiguous()

        # Resize image to IMAGE_SIZE with bilinear interpolation.
        img_res = nn.functional.interpolate(
            img.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", align_corners=False
        ).squeeze(0)

        # Resize mask to IMAGE_SIZE with nearest‑neighbor interpolation.
        mask_res = nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(), size=IMAGE_SIZE, mode="nearest"
        ).squeeze(0).squeeze(0).long()

        resized_images.append(img_res)
        resized_masks.append(mask_res)

    return torch.stack(resized_images), torch.stack(resized_masks)


def resize_batch(images, masks, size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize a batch of images and masks.

    Parameters
    ----------
    images : torch.Tensor
        Shape ``[B, 3, H, W]``.
    masks : torch.Tensor
        Shape ``[B, H, W]`` with integer class indices.
    size : tuple[int, int]
        Desired output size ``(height, width)``.
    """
    # Image resize – bilinear
    images_resized = nn.functional.interpolate(images, size=size, mode="bilinear", align_corners=False)
    # Mask resize – nearest
    masks_resized = nn.functional.interpolate(
        masks.unsqueeze(1).float(), size=size, mode="nearest"
    ).squeeze(1).long()
    return images_resized, masks_resized

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets & loaders
    train_dataset = OpeningSegmentationDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = OpeningSegmentationDataset(VAL_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Model
    model = UNetResNet34(
        in_channels=3, out_channels=3, pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_mean_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ---------- Train ----------
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Resize to model input size
            images_res, masks_res = resize_batch(images, masks, IMAGE_SIZE)

            optimizer.zero_grad()
            logits = model(images_res)
            loss = criterion(logits, masks_res)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

        train_loss = train_loss_total / train_batches

        # ---------- Validate ----------
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        total_intersection = [0, 0, 0]
        total_union = [0, 0, 0]
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                images_res, masks_res = resize_batch(images, masks, IMAGE_SIZE)

                logits = model(images_res)
                loss = criterion(logits, masks_res)
                val_loss_total += loss.item()
                val_batches += 1

                preds = logits.argmax(dim=1)
                for cls in range(3):
                    pred_cls = preds == cls
                    mask_cls = masks_res == cls
                    intersection = (pred_cls & mask_cls).sum().item()
                    union = (pred_cls | mask_cls).sum().item()
                    total_intersection[cls] += intersection
                    total_union[cls] += union

        val_loss = val_loss_total / val_batches

        ious = []
        for cls in range(3):
            if total_union[cls] == 0:
                ious.append(float("nan"))
            else:
                ious.append(total_intersection[cls] / total_union[cls])

        background_iou, door_iou, window_iou = ious
        # Compute mean IoU ignoring NaNs
        valid_ious = [iou for iou in ious if not math.isnan(iou)]
        if valid_ious:
            mean_iou_overall = sum(valid_ious) / len(valid_ious)
        else:
            mean_iou_overall = float("nan")

        # ---------- Checkpoint ----------
        if mean_iou_overall > best_mean_iou:
            best_mean_iou = mean_iou_overall
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(
                f"[Epoch {epoch}] New best mean IoU: {best_mean_iou:.4f} – model checkpoint saved."
            )

        # ---------- Logging ----------
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Bg IoU: {background_iou:.4f} | "
            f"Door IoU: {door_iou:.4f} | "
            f"Window IoU: {window_iou:.4f} | "
            f"Mean IoU: {mean_iou_overall:.4f}"
        )


if __name__ == "__main__":
    main()

# End of file
