"""Shared image geometry helpers for CrowdSim visual policies."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_with_letterbox(
    images: torch.Tensor,
    size: int = 224,
    *,
    mode: str = "bilinear",
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Resize NCHW tensors without changing aspect ratio, then square-pad.

    This mirrors FLUX/NavDP preprocessing: the longer image side becomes
    ``size`` and the shorter side is padded equally on both sides.  Odd padding
    is placed on the bottom/right so the output is always exactly square.
    """
    if images.dim() != 4:
        raise ValueError(f"Expected NCHW tensor, got {tuple(images.shape)}")
    target = int(size)
    if target <= 0:
        raise ValueError(f"Letterbox size must be positive, got {target}")
    height, width = images.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(height, width)}")
    if (height, width) == (target, target):
        return images

    scale = target / max(height, width)
    resized_height = min(target, max(1, int(round(height * scale))))
    resized_width = min(target, max(1, int(round(width * scale))))
    align_corners = False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=align_corners,
    )
    pad_height = target - resized_height
    pad_width = target - resized_width
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    return F.pad(resized, (left, right, top, bottom), value=float(pad_value))
