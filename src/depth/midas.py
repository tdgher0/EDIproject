from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

_MIDAS: tuple | None = None


def load_midas():
    """Load and cache MiDaS Small. Output is relative depth, not metres."""
    global _MIDAS
    if _MIDAS is not None:
        return _MIDAS

    model = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        trust_repo=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True,
    )

    _MIDAS = (model, transforms.small_transform, device)
    return _MIDAS


def predict_depth(
    image: np.ndarray,
    model,
    transform,
    device: torch.device,
) -> np.ndarray:
    """Relative depth at the input resolution. `image` must be OpenCV BGR."""
    input_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()


def save_depth_visualization(
    depth: np.ndarray,
    output_path: str | Path,
) -> None:
    """Normalize relative depth for display only; not for metric calculations."""
    depth_image = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), depth_image)


def save_raw_depth(
    depth: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, depth)
