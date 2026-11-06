"""Model backends. Each backend exposes:

    size, num_frames, tubelet, patch, grid          input geometry
    forward(x, mask) -> (pred, target)               x: (B, T, 3, S, S) normalised
                                                     mask: bool (N,), True = target
                                                     pred/target: (B, N_target, D)
    token_errors(x, mask) -> (B, N) float32, NaN at context tokens

IMPORTANT (information leakage): the context encoder must NOT see the target
tokens. Both backends here satisfy this by construction (masked tokens are
dropped before the encoder); scripts/selftest.py verifies it empirically.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .preprocess import IMAGENET_MEAN, IMAGENET_STD


class _Base:
    size: int
    num_frames: int
    tubelet: int
    patch: int
    error_kind: str

    @property
    def grid(self) -> Tuple[int, int, int]:
        return (self.num_frames // self.tubelet, self.size // self.patch, self.size // self.patch)

    @property
    def n_tokens(self) -> int:
        t, h, w = self.grid
        return t * h * w

    def forward(self, x, mask):  # pragma: no cover
        raise NotImplementedError

    def _per_token(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def token_errors(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred, target = self.forward(x, mask)
        e = self._per_token(pred.float(), target.float())  # (B, Nt)
        out = torch.full((x.shape[0], self.n_tokens), float("nan"), device=e.device)
        out[:, mask.to(e.device)] = e
        return out.cpu()


# ----------------------------------------------------------------------------
# VideoMAE (pixel reconstruction) via HuggingFace transformers
# ----------------------------------------------------------------------------
class VideoMAEBackend(_Base):
    """Pixel-level probe. Uses the pretraining checkpoint *with decoder*
    (default MCG-NJU/videomae-base, Kinetics-400 self-supervised, no labels).
    Error = per-tubelet MSE in the same (patch-normalised) pixel space the model
    was trained on (norm_pix_loss)."""

    error_kind = "pixel_mse"

    def __init__(self, model_id: str = "MCG-NJU/videomae-base", device: str = "cuda",
                 amp: bool = True):
        from transformers import VideoMAEForPreTraining

        self.device = device
        self.amp = amp
        self.model = VideoMAEForPreTraining.from_pretrained(model_id).to(device).eval()
        cfg = self.model.config
        self.size = cfg.image_size
        self.num_frames = cfg.num_frames
        self.tubelet = cfg.tubelet_size
        self.patch = cfg.patch_size
        self.norm_pix = getattr(cfg, "norm_pix_loss", True)
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD

    def _targets(self, x: torch.Tensor) -> torch.Tensor:
        """Replicates VideoMAEForPreTraining's label construction -> (B, N, P)."""
        m = torch.tensor(self.mean, device=x.device).view(1, 1, 3, 1, 1)
        s = torch.tensor(self.std, device=x.device).view(1, 1, 3, 1, 1)
        fr = x.float() * s + m
        B, T, C, H, W = fr.shape
        t, p = self.tubelet, self.patch
        fr = fr.view(B, T // t, t, C, H // p, p, W // p, p)
        fr = fr.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
        if self.norm_pix:
            fr = fr.view(B, (T // t) * (H // p) * (W // p), t * p * p, C)
            fr = (fr - fr.mean(dim=-2, keepdim=True)) / (fr.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            return fr.view(B, (T // t) * (H // p) * (W // p), t * p * p * C)
        return fr.view(B, (T // t) * (H // p) * (W // p), t * p * p * C)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_loss: bool = False):
        x = x.to(self.device)
        B = x.shape[0]
        bm = mask.to(self.device).unsqueeze(0).expand(B, -1).contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            out = self.model(pixel_values=x, bool_masked_pos=bm)
        pred = out.logits.float()                                   # (B, Nt, P)
        target = self._targets(x)[bm].reshape(B, -1, pred.shape[-1])
        if return_loss:
            return pred, target, out.loss
        return pred, target

    def _per_token(self, pred, target):
        return ((pred - target) ** 2).mean(-1)


# ----------------------------------------------------------------------------
# V-JEPA 2 (feature prediction) via torch.hub (facebookresearch/vjepa2)
# ----------------------------------------------------------------------------
class VJEPA2Backend(_Base):
    """Feature-level probe. Context encoder sees ONLY context tokens (masks passed
    to the encoder); the predictor predicts target-token features; the target is
    the layer-normed encoder output on the full clip (as in V-JEPA training).
    Error = per-token L1 (V-JEPA's training loss).

    Loading (fixes two upstream hazards found on 2026-11-06):
      * upstream `main` hard-codes VJEPA_BASE_URL = "http://localhost:8300", so the hub's
        own pretrained download fails -> we build the architecture with pretrained=False
        and download the checkpoint ourselves from the official URL (or a local file);
      * upstream loads weights with strict=False -> key mismatches would silently leave
        a RANDOM predictor -> we load ourselves and fail loudly on missing keys.
      * context encoder = checkpoint['encoder'] (what the predictor was trained on),
        target encoder = checkpoint['target_encoder'] (EMA); falls back if absent.
    Pin the code version with repo="facebookresearch/vjepa2:<commit-sha>".
    The HuggingFace `VJEPA2Model` is deliberately NOT used: its predictor reads
    encoder outputs computed on the full (unmasked) clip, which leaks target info.
    """

    error_kind = "feature_l1"
    OFFICIAL_URL = "https://dl.fbaipublicfiles.com/vjepa2/{file}.pt"
    ARCH_FILE = {"vjepa2_vit_large": "vitl", "vjepa2_vit_huge": "vith", "vjepa2_vit_giant": "vitg",
                 "vjepa2_vit_giant_384": "vitg-384"}

    def __init__(self, hub_name: str = "vjepa2_vit_large", repo: str = "facebookresearch/vjepa2",
                 weights: str = None, size: int = 256, num_frames: int = 16, device: str = "cuda",
                 amp: bool = True):
        import copy

        res = torch.hub.load(repo, hub_name, pretrained=False, trust_repo=True)
        if not (isinstance(res, (tuple, list)) and len(res) >= 2):
            raise RuntimeError(
                f"torch.hub.load('{repo}', '{hub_name}') returned {type(res)}; expected (encoder, predictor).")
        enc, pred = res[0], res[1]
        weights = weights or self.OFFICIAL_URL.format(file=self.ARCH_FILE[hub_name])
        if weights.startswith("http"):
            sd = torch.hub.load_state_dict_from_url(weights, map_location="cpu")
        else:
            sd = torch.load(weights, map_location="cpu")
        print(f"[vjepa2] code={repo} weights={weights} checkpoint keys={list(sd.keys())}")
        tgt_key = "target_encoder" if "target_encoder" in sd else "encoder"
        ctx_key = "encoder" if "encoder" in sd else tgt_key
        tgt_enc = copy.deepcopy(enc)
        _load_checked(enc, sd[ctx_key], f"context encoder ({ctx_key})")
        _load_checked(tgt_enc, sd[tgt_key], f"target encoder ({tgt_key})")
        _load_checked(pred, sd["predictor"], "predictor")
        self.encoder = enc.to(device).eval()
        self.target_encoder = tgt_enc.to(device).eval()
        self.predictor = pred.to(device).eval()
        self.weights_info = dict(repo=repo, weights=weights, ctx_key=ctx_key, tgt_key=tgt_key)
        self.device, self.amp = device, amp
        self.size, self.num_frames = size, num_frames
        pe = getattr(self.encoder, "patch_embed", None)
        self.patch = int(getattr(self.encoder, "patch_size", None) or _first(getattr(pe, "patch_size", 16)))
        self.tubelet = int(getattr(self.encoder, "tubelet_size", None) or getattr(pe, "tubelet_size", 2))
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD

    def _idx(self, mask: torch.Tensor, B: int):
        tgt = torch.nonzero(mask, as_tuple=False).squeeze(1)
        ctx = torch.nonzero(~mask, as_tuple=False).squeeze(1)
        tgt = tgt.to(self.device).unsqueeze(0).expand(B, -1).contiguous()
        ctx = ctx.to(self.device).unsqueeze(0).expand(B, -1).contiguous()
        return ctx, tgt

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x.to(self.device).permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, T, H, W)
        B = x.shape[0]
        ctx_idx, tgt_idx = self._idx(mask, B)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            full = self.target_encoder(x)                            # (B, N, D)
            ctx = self.encoder(x, masks=[ctx_idx])                   # context only
            pred = self.predictor(ctx, ctx_idx, tgt_idx)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        full = F.layer_norm(full.float(), (full.shape[-1],))
        target = torch.gather(full, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
        pred = pred.float()
        if pred.shape != target.shape:
            raise RuntimeError(f"predictor output {tuple(pred.shape)} != target {tuple(target.shape)}; "
                               "check predictor API / output projection.")
        return pred, target

    def _per_token(self, pred, target):
        return (pred - target).abs().mean(-1)


def _clean_keys(sd):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def _load_checked(model, sd, name):
    """load_state_dict(strict=False) but fail on missing weights (pos_embed is
    expected to be unused because the model uses RoPE)."""
    r = model.load_state_dict(_clean_keys(sd), strict=False)
    missing = [k for k in r.missing_keys if "pos_embed" not in k]
    unexpected = [k for k in r.unexpected_keys if "pos_embed" not in k]
    print(f"[vjepa2] {name}: missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}"
          + (f" (non-pos_embed missing: {missing[:5]})" if missing else ""))
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} weights missing from checkpoint, e.g. {missing[:5]} "
                           "-> the model would run partly RANDOM. Check code version vs checkpoint.")
    if unexpected:
        print(f"[vjepa2] warning {name}: unexpected keys ignored, e.g. {unexpected[:5]}")


def _first(v):
    return v[0] if isinstance(v, (tuple, list)) else v


def build_backend(name: str, **kw) -> _Base:
    if name == "videomae":
        return VideoMAEBackend(model_id=kw.get("model_id") or "MCG-NJU/videomae-base",
                               device=kw.get("device", "cuda"), amp=kw.get("amp", True))
    if name == "vjepa2":
        return VJEPA2Backend(hub_name=kw.get("hub_name") or "vjepa2_vit_large",
                             repo=kw.get("hub_repo") or "facebookresearch/vjepa2",
                             weights=kw.get("weights"),
                             size=kw.get("size") or 256, num_frames=kw.get("num_frames") or 16,
                             device=kw.get("device", "cuda"), amp=kw.get("amp", True))
    raise ValueError(name)
