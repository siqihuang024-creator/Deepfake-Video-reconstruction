"""Frames (uint8, T x H x W x 3) -> model input tensors."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_short(x: torch.Tensor, size: int) -> torch.Tensor:
    """x: (T, 3, H, W) float in [0,1]."""
    _, _, h, w = x.shape
    s = size / min(h, w)
    nh, nw = max(size, int(round(h * s))), max(size, int(round(w * s)))
    return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)


def crops(x: torch.Tensor, size: int, mode: str) -> List[torch.Tensor]:
    """Square crops along the long side. mode: 'center' or 'three' (start/center/end,
    together covering the whole frame for ~16:9 video)."""
    _, _, h, w = x.shape
    if h >= w:
        offs = [(h - size) // 2] if mode == "center" else [0, (h - size) // 2, h - size]
        return [x[:, :, o:o + size, :size] for o in offs]
    offs = [(w - size) // 2] if mode == "center" else [0, (w - size) // 2, w - size]
    return [x[:, :, :size, o:o + size] for o in offs]


def frames_to_inputs(frames: np.ndarray, size: int, crop_mode: str,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Returns (n_crops, T, 3, size, size) normalised, plus the unnormalised crops
    (same shape, [0,1]) for computing covariates."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    x = resize_short(x, size)
    cs = crops(x, size, crop_mode)
    raw = torch.stack(cs, 0)
    m = torch.tensor(mean).view(1, 1, 3, 1, 1)
    s = torch.tensor(std).view(1, 1, 3, 1, 1)
    return (raw - m) / s, raw


def covariates(raw: torch.Tensor) -> dict:
    """Content-difficulty covariates from unnormalised crops (n, T, 3, H, W):
    motion  = mean |frame_{t+1} - frame_t|
    texture = mean spatial gradient magnitude
    Used later to calibrate scores for scene difficulty and to check confounds."""
    g = raw.mean(2)  # grey (n, T, H, W)
    motion = (g[:, 1:] - g[:, :-1]).abs().mean().item()
    gx = (g[..., :, 1:] - g[..., :, :-1]).abs().mean()
    gy = (g[..., 1:, :] - g[..., :-1, :]).abs().mean()
    return {"motion": motion, "texture": float((gx + gy).item() / 2)}
