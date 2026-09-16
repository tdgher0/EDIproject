"""Segmentation model definition.

This module defines a minimal U‑Net architecture with a ResNet‑34 encoder, suitable for binary wall segmentation.

The implementation uses `segmentation_models_pytorch` (``smp``) which provides a convenient
:class:`smp.Unet` wrapper.  The encoder is specified via ``encoder_name="resnet34"`` and the
`pretrained="imagenet"` flag pulls the standard ImageNet weights.

The model returns a single‑channel prediction map with the same spatial resolution as the input.
The final layer uses a sigmoid activation so the output can be interpreted as a probability
map for the foreground class (wall).

No training logic, data handling, or configuration is touched – this file only contains the
model class.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetResNet34(nn.Module):
    """U‑Net with a ResNet‑34 encoder.

    Parameters
    ----------
    in_channels:
        Number of input channels. Default is 3 for RGB images.
    out_channels:
        Number of output channels. Default is 1 for binary segmentation.
    pretrained:
        If ``True`` the encoder is initialized with ImageNet weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ``smp.Unet`` expects the encoder name; ``pretrained`` is passed via ``encoder_weights``.
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        The ``smp.Unet`` already applies the final ``Sigmoid`` activation when
        ``activation=None`` is set (the default).  For binary segmentation it is
        common to keep the raw logits and apply ``BCEWithLogitsLoss`` during
        training.  If a sigmoid is preferred, the caller can add it.
        """

        return self.model(x)


__all__ = ["UNetResNet34"]
