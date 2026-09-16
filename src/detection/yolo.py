from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO


class YOLODetector:
    """YOLO instance-segmentation wrapper for object detection."""

    def __init__(
        self,
        model_path: str = "yolo11n-seg.pt",
        confidence: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        self.confidence = confidence

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def predict(self, image: np.ndarray) -> list[dict]:
        """
        Run YOLO segmentation on an image.

        Returns:
            List of detections containing class information, bounding box,
            confidence, and a full-resolution boolean instance mask.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")

        if image.ndim != 3:
            raise ValueError("image must have shape (H, W, C)")

        height, width = image.shape[:2]

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        detections: list[dict] = []

        if result.boxes is None or result.masks is None:
            return detections

        masks = result.masks.data

        for i, (box, confidence, class_id) in enumerate(
            zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
        ):
            mask = masks[i].detach().cpu().numpy()

            mask = (
                torch.from_numpy(mask[None, None])
                .float()
            )
            mask = torch.nn.functional.interpolate(
                mask,
                size=(height, width),
                mode="nearest",
            )[0, 0].numpy()

            detections.append(
                {
                    "class_id": int(class_id.item()),
                    "class_name": str(
                        self.model.names[int(class_id.item())]
                    ),
                    "confidence": float(confidence.item()),
                    "bbox": tuple(
                        int(round(value))
                        for value in box.detach().cpu().tolist()
                    ),
                    "mask": mask.astype(bool),
                }
            )

        return detections
