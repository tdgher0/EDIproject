from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.segmentation.model import UNetResNet34

__all__ = ["predict_wall_mask"]

# Production threshold selected from downstream wall-area MAE, not segmentation IoU.
WALL_PROBABILITY_THRESHOLD = 0.8

_IMAGE_SIZE_CACHE: dict[str, tuple[int, int]] = {}
_MODEL: torch.nn.Module | None = None
_CHECKPOINT_PATH: str | None = None
_DEVICE: torch.device | None = None


def _load_image_size(config_path: str | Path = "configs/model_config.yaml") -> tuple[int, int]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cache_key = str(path.resolve())
    cached = _IMAGE_SIZE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        size = cfg["train"]["image_size"]
    except (TypeError, KeyError) as exc:
        raise ValueError("Could not find 'image_size' in the configuration.") from exc

    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("Could not find 'image_size' in the configuration.")

    result = (int(size[0]), int(size[1]))
    _IMAGE_SIZE_CACHE[cache_key] = result
    return result


def _load_model(checkpoint_path: str | os.PathLike) -> torch.nn.Module:
    global _MODEL, _CHECKPOINT_PATH, _DEVICE
    checkpoint_key = str(checkpoint_path)
    if _MODEL is not None and _CHECKPOINT_PATH == checkpoint_key:
        return _MODEL

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ImageNet encoder weights are unused; the trained checkpoint replaces them.
    model = UNetResNet34(in_channels=3, out_channels=1, pretrained=False)
    model.to(_DEVICE)

    checkpoint = torch.load(checkpoint_key, map_location=_DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL = model
    _CHECKPOINT_PATH = checkpoint_key
    return _MODEL


def predict_wall_mask(
    image_path: str | os.PathLike,
    checkpoint_path: str | os.PathLike = "models/checkpoints/best_model.pt",
) -> np.ndarray:
    """Return a boolean wall mask at the original image resolution."""
    model = _load_model(checkpoint_path)

    img = Image.open(str(image_path)).convert("RGB")
    orig_w, orig_h = img.size

    img_np = np.asarray(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(_DEVICE)

    target_h, target_w = _load_image_size()
    img_tensor = F.interpolate(
        img_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        binary = probs > WALL_PROBABILITY_THRESHOLD

    mask = F.interpolate(binary.float(), size=(orig_h, orig_w), mode="nearest")
    return mask.squeeze().cpu().numpy().astype(bool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wall segmentation inference.")
    parser.add_argument("image", type=str, help="Path to an input image.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save the output mask (PNG).",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pt",
        help="Path to the model checkpoint.",
    )
    args = parser.parse_args()

    mask = predict_wall_mask(args.image, args.checkpoint)
    print(f"Predicted mask shape: {mask.shape}, dtype={mask.dtype}")

    if args.output:
        out_path = Path(args.output)
        Image.fromarray((mask.astype(np.uint8)) * 255).save(out_path)
        print(f"Mask saved to {out_path}")
