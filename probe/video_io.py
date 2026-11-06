"""Video metadata (ffprobe), clip planning and frame decoding.

Decoding backends: PyAV (preferred, timestamp-accurate) -> OpenCV (fallback,
available in the cluster container).
"""
from __future__ import annotations

import json
import math
import subprocess
from typing import List, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
def _rate(x: Optional[str]) -> Optional[float]:
    try:
        n, d = x.split("/")
        n, d = float(n), float(d)
        return n / d if d else None
    except Exception:
        return None


def ffprobe_meta(path: str) -> dict:
    """Return codec / size / fps / duration / bitrate of the first video stream."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,bit_rate,pix_fmt"
        ":format=duration,bit_rate,size",
        "-of", "json", path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {res.stderr.strip()}")
    out = json.loads(res.stdout)
    s = out["streams"][0]
    f = out.get("format", {})
    nb = s.get("nb_frames")
    fps = _rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate"))
    duration = float(f["duration"]) if f.get("duration") not in (None, "N/A") else None
    br = s.get("bit_rate") if s.get("bit_rate") not in (None, "N/A") else f.get("bit_rate")
    br = int(br) if br not in (None, "N/A") else None
    w, h = int(s["width"]), int(s["height"])
    meta = dict(
        codec=s.get("codec_name"),
        width=w,
        height=h,
        fps=round(fps, 4) if fps else None,
        nb_frames=int(nb) if (nb and str(nb).isdigit()) else None,
        duration=round(duration, 4) if duration else None,
        bitrate=br,
        pix_fmt=s.get("pix_fmt"),
        size_bytes=int(f["size"]) if f.get("size") else None,
    )
    # bits per pixel per frame: a crude "compression strength" covariate
    if br and fps:
        meta["bpp"] = round(br / (w * h * fps), 6)
    else:
        meta["bpp"] = None
    return meta


# ----------------------------------------------------------------------------
# Clip planning
# ----------------------------------------------------------------------------
def plan_clips(duration: float, clip_fps: float, num_frames: int,
               skip_start: float, max_clips: int, end_margin: float = 0.05) -> List[List[float]]:
    """Return a list of clips, each a list of `num_frames` timestamps (seconds).

    Clips are evenly spread over [skip_start, duration - end_margin].
    `skip_start` matters for I2V data (e.g. CoCoVideo), where the first frame of
    the fake is the real first frame.
    Returns [] if the video is too short.
    """
    span = (num_frames - 1) / clip_fps
    lo = skip_start
    hi = duration - end_margin - span
    if hi < lo:
        # try without skip_start if the video is just a bit short
        if duration - end_margin - span >= 0:
            lo = hi = max(0.0, duration - end_margin - span)
        else:
            return []
    if max_clips <= 1 or hi - lo < 1e-6:
        starts = [(lo + hi) / 2]
    else:
        # at most 50 % overlap between neighbouring clips
        n_fit = int(math.floor((hi - lo) / (span * 0.5))) + 1
        n = max(1, min(max_clips, n_fit))
        starts = list(np.linspace(lo, hi, n)) if n > 1 else [(lo + hi) / 2]
    return [[s + i / clip_fps for i in range(num_frames)] for s in starts]


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------
def _decode_av(path: str, ts: List[float]) -> List[np.ndarray]:
    import av  # noqa

    out: List[Optional[np.ndarray]] = [None] * len(ts)
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        t0 = float(st.start_time * st.time_base) if st.start_time is not None else 0.0
        i = 0
        prev, prev_t = None, None
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            t = float(fr.pts * st.time_base) - t0
            while i < len(ts) and t >= ts[i]:
                pick = prev if (prev is not None and abs(prev_t - ts[i]) < abs(t - ts[i])) else fr
                out[i] = pick.to_ndarray(format="rgb24")
                i += 1
            if i >= len(ts):
                break
            prev, prev_t = fr, t
        while i < len(ts):  # timestamps past the last frame -> last frame
            if prev is None:
                raise RuntimeError(f"no frames decoded from {path}")
            out[i] = prev.to_ndarray(format="rgb24")
            i += 1
    return out  # type: ignore


def _decode_cv2(path: str, ts: List[float], fps_hint: Optional[float]) -> List[np.ndarray]:
    import cv2  # noqa

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    if not fps or fps <= 0:
        raise RuntimeError(f"unknown fps for {path}")
    want = [int(round(t * fps)) for t in ts]
    out: List[Optional[np.ndarray]] = [None] * len(ts)
    idx, j, last = 0, 0, None
    while j < len(want):
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
        while j < len(want) and want[j] == idx:
            out[j] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            j += 1
        idx += 1
    cap.release()
    if last is None:
        raise RuntimeError(f"no frames decoded from {path}")
    for k in range(len(out)):
        if out[k] is None:
            out[k] = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
    return out  # type: ignore


_BACKEND = None


def decode_frames_at(path: str, timestamps: List[float], fps_hint: Optional[float] = None) -> np.ndarray:
    """Decode frames nearest to the given (sorted or unsorted) timestamps.
    Returns uint8 array (N, H, W, 3) in the order of `timestamps`."""
    global _BACKEND
    order = np.argsort(timestamps)
    ts_sorted = [float(timestamps[k]) for k in order]
    if _BACKEND is None:
        try:
            import av  # noqa
            _BACKEND = "av"
        except ImportError:
            _BACKEND = "cv2"
    frames = _decode_av(path, ts_sorted) if _BACKEND == "av" else _decode_cv2(path, ts_sorted, fps_hint)
    res = [None] * len(timestamps)
    for pos, k in enumerate(order):
        res[k] = frames[pos]
    return np.stack(res, 0)


def decoder_backend() -> str:
    try:
        import av  # noqa
        return "av"
    except ImportError:
        return "cv2"
