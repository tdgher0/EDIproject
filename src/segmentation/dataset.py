"""PyTorch Dataset for binary wall segmentation.

The dataset expects a directory layout with an ``images`` folder containing RGB JPEGs and a ``masks`` folder containing the corresponding single‑channel PNG masks.  The two folders must be siblings and the mask filenames must match the image filenames (excluding the file extension).

Example root structure::

    root/
        images/
            img-000001.jpg
            img-000002.jpg
            ...
        masks/
            img-000001.png
            img-000002.png
            ...

The class implements the minimal interface required by the training script: it returns a tuple ``(image, mask)`` where ``image`` is a float tensor of shape ``[3, H, W]`` in the ``[0, 1]`` range and ``mask`` is a float tensor of shape ``[1, H, W]`` with values 0.0 or 1.0.

An optional ``transform`` callable can be passed to apply custom augmentation or preprocessing to the pair ``(image, mask)`` after the default conversion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class WallSegmentationDataset(Dataset):
    """Dataset for binary wall segmentation.

    Parameters
    ----------
    root:
        Path to a directory that contains an ``images`` subdirectory and a ``masks`` subdirectory.
    transform:
        Optional callable applied to the ``(image, mask)`` pair after conversion.  It should
        accept and return the same types.
    """

    def __init__(self, root: str | Path, transform: Optional[Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]] = None) -> None:
        super().__init__()
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.transform = transform

        # Gather image paths; assume masks exist with the same stem.
        self.image_paths = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No image files found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image {img_path}")

        # Load image and mask using Pillow.
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # single channel

        # Convert to tensors.
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0

        # Rearrange to channel first.
        image_tensor = image_tensor.permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
        mask_tensor = mask_tensor.unsqueeze(0)  # [H,W] -> [1,H,W]

        # Optional transform.
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor


__all__ = ["WallSegmentationDataset"]
