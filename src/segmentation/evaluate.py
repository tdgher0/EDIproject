"""Evaluation script for SUN RGB‑D wall segmentation.

The script loads the test set, restores a trained UNetResNet34 model, and
computes the following metrics over the entire test split:

* IoU (Intersection‑over‑Union)
* Dice coefficient
* Precision
* Recall

The implementation deliberately follows the same preprocessing steps that were
used during training – a 512×512 resize with bilinear interpolation for
images and nearest‑neighbour interpolation for masks – so that the results are
consistent.

The script is intentionally minimal and beginner‑friendly.  It does not
train the model, save predictions, or create visualisations – it only reports
the metrics.
"""

from __future__ import annotations

import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.segmentation.dataset import WallSegmentationDataset
from src.segmentation.model import UNetResNet34

# ---------------------------------------------------------------------------
# Helper: metrics for a single prediction/target pair
# ---------------------------------------------------------------------------

def _binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, float]:
    """Compute IoU, Dice, Precision and Recall for a single binary mask.

    Parameters
    ----------
    pred:
        Binary mask (dtype=torch.uint8/int/float) with values 0 or 1.
    target:
        Binary ground‑truth mask.

    Returns
    -------
    iou, dice, precision, recall:
        Metric values as floats in [0, 1].  The function guards against
        division‑by‑zero by defining the metric as 1.0 when both the
        prediction and the target are empty.
    """
    # Ensure integer masks for set operations
    pred = pred.int()
    target = target.int()

    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()

    pred_sum = pred.sum().item()
    target_sum = target.sum().item()

    # IoU
    if union > 0:
        iou = intersection / union
    else:
        iou = 1.0

    # Dice
    if pred_sum + target_sum > 0:
        dice = 2.0 * intersection / (pred_sum + target_sum)
    else:
        dice = 1.0

    # Precision
    if pred_sum > 0:
        precision = intersection / pred_sum
    else:
        precision = 1.0

    # Recall
    if target_sum > 0:
        recall = intersection / target_sum
    else:
        recall = 1.0

    return iou, dice, precision, recall

# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main() -> None:
    # Load configuration -----------------------------------------------------
    cfg_path = "configs/model_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Dataset & DataLoader ---------------------------------------------------
    test_dir = cfg["data"]["test"]

    # Pre‑processing identical to training
    transform = lambda img, mask: (
        F.interpolate(img.unsqueeze(0), size=tuple(cfg["train"]["image_size"]), mode="bilinear", align_corners=False).squeeze(0),
        F.interpolate(mask.unsqueeze(0), size=tuple(cfg["train"]["image_size"]), mode="nearest").squeeze(0),
    )

    test_ds = WallSegmentationDataset(root=test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # Device & model --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    model = UNetResNet34(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
        pretrained=model_cfg.get("pretrained", True),
    )
    model.to(device)

    ckpt_path = os.path.join("models", "checkpoints", "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Accumulators ----------------------------------------------------------
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    num_samples = 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)  # shape: [B, 1, H, W]
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            target = (masks > 0.5).int()

            iou, dice, precision, recall = _binary_metrics(preds[0], target[0])
            total_iou += iou
            total_dice += dice
            total_precision += precision
            total_recall += recall
            num_samples += 1

    # Compute averages -----------------------------------------------------
    if num_samples == 0:
        print("No test samples found.")
        return

    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples

    # Summary --------------------------------------------------------------
    print(f"Test samples: {num_samples}")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Dice: {avg_dice:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")


if __name__ == "__main__":
    main()
