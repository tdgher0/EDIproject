from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.segmentation.model import UNetResNet34

# Matches train_opening.py: 512×512 bilinear input, argmax over 3 classes.
_OPENING_IMAGE_SIZE = (512, 512)

_MODEL: torch.nn.Module | None = None
_CHECKPOINT_PATH: str | None = None
_DEVICE: torch.device | None = None


def _load_model(checkpoint_path: str | Path) -> torch.nn.Module:
    global _MODEL, _CHECKPOINT_PATH, _DEVICE
    if _MODEL is not None and _CHECKPOINT_PATH == str(checkpoint_path):
        return _MODEL

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet34(in_channels=3, out_channels=3, pretrained=False)
    model.to(_DEVICE)
    state_dict = torch.load(str(checkpoint_path), map_location=_DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    _MODEL = model
    _CHECKPOINT_PATH = str(checkpoint_path)
    return _MODEL


def predict_opening_masks(
    image_path: str | Path,
    checkpoint_path: str | Path = "models/checkpoints/opening_augmented_best_model.pt",
) -> tuple[np.ndarray, np.ndarray]:
    """Return door (class 1) and window (class 2) masks at original image size."""
    model = _load_model(checkpoint_path)

    img = Image.open(str(image_path)).convert("RGB")
    orig_w, orig_h = img.size

    img_np = np.asarray(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(_DEVICE)

    img_tensor = F.interpolate(
        img_tensor,
        size=_OPENING_IMAGE_SIZE,
        mode="bilinear",
        align_corners=False,
    )

    with torch.no_grad():
        logits = model(img_tensor)
        pred = logits.argmax(dim=1)

    pred = F.interpolate(
        pred.unsqueeze(1).float(),
        size=(orig_h, orig_w),
        mode="nearest",
    ).squeeze(1).long()

    pred_np = pred.squeeze(0).cpu().numpy()
    return pred_np == 1, pred_np == 2
