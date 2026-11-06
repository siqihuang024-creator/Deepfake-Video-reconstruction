"""Token masks over a (T', H', W') tubelet grid, flattened in (t, h, w) order
(the order used by both VideoMAE and V-JEPA patch embeddings).

A mask is a bool tensor of length N = T'*H'*W'; True = target (hidden, to be
predicted), False = visible context.

Mask spec strings (used on the command line):
    future            hide the second half in time (predict the future)
    future:0.25       hide the last 25 % of temporal slots
    tube:0.9          random spatial positions, hidden across all time (VideoMAE pretraining style)
    block:0.25        one random spatial rectangle (~25 % area), hidden across all time
                      -> "can the surroundings explain this region over time?"
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def parse_spec(spec: str) -> Tuple[str, float]:
    if ":" in spec:
        kind, v = spec.split(":", 1)
        return kind, float(v)
    defaults = {"future": 0.5, "tube": 0.9, "block": 0.25}
    if spec not in defaults:
        raise ValueError(f"unknown mask spec {spec}")
    return spec, defaults[spec]


def is_random(spec: str) -> bool:
    return parse_spec(spec)[0] in ("tube", "block")


def make_mask(spec: str, grid: Tuple[int, int, int], seed: int = 0) -> torch.Tensor:
    T, H, W = grid
    kind, v = parse_spec(spec)
    m = np.zeros((T, H, W), dtype=bool)
    rng = np.random.default_rng(seed)
    if kind == "future":
        k = max(1, int(round(T * v)))
        k = min(k, T - 1)  # keep at least one context slot
        m[T - k:] = True
    elif kind == "tube":
        n = int(round(H * W * v))
        n = min(max(n, 1), H * W - 1)
        pos = rng.choice(H * W, size=n, replace=False)
        sp = np.zeros(H * W, dtype=bool)
        sp[pos] = True
        m[:] = sp.reshape(H, W)[None]
    elif kind == "block":
        area = v * H * W
        ar = float(np.exp(rng.uniform(np.log(0.75), np.log(1.5))))
        bh = int(round(np.sqrt(area / ar)))
        bw = int(round(np.sqrt(area * ar)))
        bh, bw = min(max(bh, 1), H - 1), min(max(bw, 1), W - 1)
        y0 = int(rng.integers(0, H - bh + 1))
        x0 = int(rng.integers(0, W - bw + 1))
        m[:, y0:y0 + bh, x0:x0 + bw] = True
    else:
        raise ValueError(kind)
    return torch.from_numpy(m.reshape(-1))


def mask_to_pixels(mask: torch.Tensor, grid: Tuple[int, int, int], tubelet: int, patch: int) -> torch.Tensor:
    """Token mask -> pixel mask (T, H, W) bool, for leakage tests / visualisation."""
    T, H, W = grid
    m = mask.view(T, H, W)
    m = m.repeat_interleave(tubelet, 0).repeat_interleave(patch, 1).repeat_interleave(patch, 2)
    return m
