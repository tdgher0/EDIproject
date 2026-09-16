"""PyTorch Dataset for the 3‑class SUN RGB‑D opening segmentation dataset.

The dataset is expected to follow the layout::

    root/
        images/
            img-000001.jpg
            ...
        masks/
            img-000001.png
            ...

Mask encoding is 0 = background, 1 = door, 2 = window.

The dataset loads images as RGB, converts them to float32 tensors in [0, 1] with shape ``[3, H, W]`` and masks to ``torch.long`` tensors with shape ``[H, W]``. It validates that mask values are only ``{0, 1, 2}`` and that the spatial dimensions of image and mask match.

An optional ``transform`` callable can be supplied which receives the ``(image_tensor, mask_tensor)`` and should return the transformed pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["OpeningSegmentationDataset"]


class OpeningSegmentationDataset(Dataset):
    """Dataset for SUN RGB‑D opening segmentation.

    Parameters
    ----------
    root : str | Path
        Root directory containing the ``images`` and ``masks`` subfolders.
    transform : callable, optional
        Optional callable that receives ``(image_tensor, mask_tensor)`` and
        returns a transformed pair.  The transform should preserve the tensor
        shapes and types.
    """

    def __init__(self, root: str | Path, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform

        image_dir = self.root / "images"
        mask_dir = self.root / "masks"

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        # Find all .jpg files in the images folder.
        image_paths = sorted(image_dir.glob("*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")

        samples: List[Tuple[Path, Path]] = []
        for img_path in image_paths:
            stem = img_path.stem
            mask_path = mask_dir / f"{stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for image {img_path.name}: expected {mask_path.name}"
                )
            samples.append((img_path, mask_path))

        self.samples = samples

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load image as RGB
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_np = np.asarray(img_rgb, dtype=np.float32) / 255.0
        # HWC -> CHW
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)))

        # Load mask without scaling
        with Image.open(mask_path) as msk:
            mask_np = np.asarray(msk, dtype=np.uint8)

        # Validate mask values
        unique_vals = np.unique(mask_np)
        if not set(unique_vals).issubset({0, 1, 2}):
            raise ValueError(
                f"Mask {mask_path.name} contains values {unique_vals.tolist()}, expected only 0, 1, 2"
            )

        # Validate spatial dimensions
        if img_np.shape[:2] != mask_np.shape:
            raise ValueError(
                f"Size mismatch between image {img_path.name} {img_np.shape[:2]} and mask {mask_path.name} {mask_np.shape}"
            )

        mask_tensor = torch.from_numpy(mask_np.astype(np.int64))

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor


# End of file
