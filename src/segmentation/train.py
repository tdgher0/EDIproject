"""Training script for SUN RGB‑D wall segmentation.

The script is intentionally minimal – it only implements what is required by the project description and the user’s specification.  It loads the configuration, builds the dataset, creates a U‑Net with a ResNet‑34 encoder, trains for the configured number of epochs, evaluates on the validation split each epoch, calculates IoU, and keeps the best checkpoint.

The code is written to be importable (so it can be used as a module) but it also runs as a standalone script via ``python src/segmentation/train.py``.
"""

from __future__ import annotations

import os
import yaml
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.segmentation.model import UNetResNet34
from src.segmentation.dataset import WallSegmentationDataset

# ---------- Utility: IoU for binary masks ---------------------------------

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """Compute IoU for a batch of binary predictions and targets.

    Parameters
    ----------
    pred:
        Predicted probabilities in the range [0, 1]. Shape: (B, 1, H, W)
    target:
        Ground‑truth binary mask. Shape: (B, 1, H, W)
    eps:
        Small value added to denominator to avoid division by zero.

    Returns
    -------
    float
        Mean IoU over the batch.
    """
    intersection = (pred * target).sum(dim=[1, 2, 3])
    union = (pred + target).sum(dim=[1, 2, 3]) - intersection
    return (intersection / (union + eps)).mean().item()

# ---------- Main training loop --------------------------------------------

def main() -> None:
    # Load config ------------------------------------------------------------
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Datasets & loaders -----------------------------------------------------
    train_ds = WallSegmentationDataset(
        cfg["data"]["train"],
        transform=lambda img, mask: (
            torch.nn.functional.interpolate(img.unsqueeze(0), size=tuple(cfg["train"]["image_size"]), mode="bilinear", align_corners=False).squeeze(0),
            torch.nn.functional.interpolate(mask.unsqueeze(0), size=tuple(cfg["train"]["image_size"]), mode="nearest").squeeze(0),
        ),
    )
    val_ds = WallSegmentationDataset(
        cfg["data"]["val"],
        transform=lambda img, mask: (
            torch.nn.functional.interpolate(img.unsqueeze(0), size=tuple(cfg["train"]["image_size"]), mode="bilinear", align_corners=False).squeeze(0),
            torch.nn.functional.interpolate(mask.unsqueeze(0), size=tuple(cfg["train"]["image_size"]), mode="nearest").squeeze(0),
        ),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # Model -----------------------------------------------------------------
    model_cfg = cfg["model"]
    model = UNetResNet34(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        pretrained=model_cfg.get("pretrained", True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss & optimizer -------------------------------------------------------
    tversky_loss = smp.losses.TverskyLoss(mode="binary", from_logits=True, alpha=0.7, beta=0.3)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    criterion = lambda logits, target: tversky_loss(logits, target) + bce_loss(logits, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])

    # Checkpoint directory ---------------------------------------------------
    ckpt_dir = os.path.join("models", "checkpoints_tversky")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_iou = -1.0

    # Training --------------------------------------------------------------
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        epoch_loss /= len(train_loader.dataset)

        # Validation --------------------------------------------------------
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)

                preds = torch.sigmoid(logits) > 0.5
                all_preds.append(preds.float())
                all_targets.append(masks)

        val_loss /= len(val_loader.dataset)
        val_iou = iou_score(torch.cat(all_preds), torch.cat(all_targets))

        print(
            f"Epoch {epoch}/{cfg['train']['epochs']}"
            f" | Train loss: {epoch_loss:.4f}"
            f" | Val loss: {val_loss:.4f}"
            f" | Val IoU: {val_iou:.4f}",
        )

        # Checkpointing -----------------------------------------------------
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print("  * New best model saved")


if __name__ == "__main__":
    main()
