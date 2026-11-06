#!/usr/bin/env python3
"""Sanity checks to run ON THE CLUSTER (interactive srun) before any full run.

    python scripts/selftest.py --backend videomae [--video some.mp4]
    python scripts/selftest.py --backend vjepa2   [--video some.mp4]

Checks
  1. model loads, input geometry, decoder backend
  2. (videomae) our per-token error reproduces HF's internal loss
  3. LEAKAGE: replacing the pixels of the target region with noise must NOT change
     the predictions (the model may only use context). Target errors should change.
  4. context matters: replacing the context with noise SHOULD change predictions.
  5. timing / peak memory for one clip with 3 crops
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe.backends import build_backend  # noqa: E402
from probe.masks import make_mask, mask_to_pixels  # noqa: E402
from probe.preprocess import frames_to_inputs  # noqa: E402
from probe.video_io import decode_frames_at, decoder_backend, ffprobe_meta, plan_clips  # noqa: E402


def synthetic(T, H=360, W=640):
    """Moving gradient + square, so there is something predictable."""
    yy, xx = np.mgrid[0:H, 0:W]
    fr = []
    for t in range(T):
        img = np.stack([(xx + 8 * t) % 256, (yy + 4 * t) % 256, np.full_like(xx, 128)], -1).astype(np.uint8)
        x0 = 50 + 10 * t
        img[100:180, x0:x0 + 80] = 255
        fr.append(img)
    return np.stack(fr)


def ok(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vjepa2", "videomae"], required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--hub-name", default=None)
    ap.add_argument("--hub-repo", default=None, help="facebookresearch/vjepa2:<commit-sha>")
    ap.add_argument("--weights", default=None, help="vjepa2 checkpoint URL or local .pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__} decoder={decoder_backend()}")

    t = time.time()
    be = build_backend(a.backend, model_id=a.model_id, hub_name=a.hub_name, hub_repo=a.hub_repo,
                       weights=a.weights, device=dev)
    print(f"1. loaded {a.backend} in {time.time() - t:.1f}s  size={be.size} frames={be.num_frames} "
          f"tubelet={be.tubelet} patch={be.patch} grid={be.grid}")

    if a.video:
        meta = ffprobe_meta(a.video)
        print(f"   video meta: {meta}")
        clip = plan_clips(meta["duration"], 8.0, be.num_frames, 0.5, 1)[0]
        frames = decode_frames_at(a.video, clip, meta["fps"])
    else:
        frames = synthetic(be.num_frames)
    x, _ = frames_to_inputs(frames, be.size, "three", be.mean, be.std)
    print(f"   input tensor {tuple(x.shape)}")
    all_pass = True

    for spec in ["future", "block:0.25"] + (["tube:0.9"] if a.backend == "videomae" else []):
        m = make_mask(spec, be.grid, seed=0)
        print(f"\n-- mask {spec}: {int(m.sum())}/{m.numel()} target tokens")
        if a.backend == "videomae":
            pred, target, loss = be.forward(x, m, return_loss=True)
            mine = ((pred - target) ** 2).mean().item()
            all_pass &= ok(abs(mine - loss.item()) / max(loss.item(), 1e-8) < 0.05,
                           f"2. per-token MSE mean {mine:.5f} vs HF loss {loss.item():.5f}")
        else:
            pred, target = be.forward(x, m)
        pix = mask_to_pixels(m, be.grid, be.tubelet, be.patch)            # (T, S, S)
        pmask = pix[None, :, None].expand_as(x)
        g = torch.Generator().manual_seed(0)
        noise = torch.randn(x.shape, generator=g)

        x_t = torch.where(pmask, noise, x)        # corrupt TARGET region
        pred_t, target_t = be.forward(x_t, m)[:2]
        dp = (pred_t - pred).abs().max().item()
        dt = (target_t - target).abs().mean().item()
        all_pass &= ok(dp < 5e-2 * (pred.abs().mean().item() + 1e-6),
                       f"3. leakage: max |Δpred| when target pixels replaced = {dp:.2e} (should be ~0); "
                       f"mean |Δtarget| = {dt:.3f} (should be >0)")

        x_c = torch.where(pmask, x, noise)        # corrupt CONTEXT
        pred_c = be.forward(x_c, m)[0]
        dc = (pred_c - pred).abs().mean().item()
        all_pass &= ok(dc > 1e-3, f"4. context sensitivity: mean |Δpred| = {dc:.3f} (should be clearly >0)")

        e = be.token_errors(x, m)
        print(f"   error mean={torch.nanmean(e).item():.4f}  per-crop={[round(torch.nanmean(r).item(), 4) for r in e]}")

    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(3):
        be.token_errors(x, make_mask("future", be.grid))
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"\n5. {(time.time() - t) / 3 * 1000:.0f} ms per clip (3 crops, 1 mask); "
          f"peak mem {torch.cuda.max_memory_allocated() / 2**30 if dev == 'cuda' else 0:.2f} GB")
    print("\nALL PASS" if all_pass else "\nSOME CHECKS FAILED -- do not run the full probe until fixed")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
