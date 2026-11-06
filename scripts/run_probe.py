#!/usr/bin/env python3
"""Run a frozen backend over every video of a manifest; write one JSON line per
(video, clip, mask, seed) with masked-prediction error statistics.

No training happens here. Labels are copied through only for later analysis.

    python scripts/run_probe.py --backend vjepa2 --manifest manifests/magic_raw.csv \
        --out results/vjepa2_magic_raw.jsonl --masks future,block:0.25 --seeds 3

    python scripts/run_probe.py --backend videomae --manifest manifests/magic_raw.csv \
        --out results/videomae_magic_raw.jsonl --masks tube:0.9,future --seeds 3
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe.backends import build_backend  # noqa: E402
from probe.masks import is_random, make_mask  # noqa: E402
from probe.preprocess import covariates, frames_to_inputs  # noqa: E402
from probe.video_io import decode_frames_at, decoder_backend, plan_clips  # noqa: E402


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not r.get("probe_error")]


def fingerprint(r):
    """Identifies the actual file behind a video_id (raw and standardised copies share
    video_ids but differ in path/size), so resume never reuses results of another file."""
    return f"{r['path']}|{r.get('size_bytes', '')}"


def stratified(rows, limit):
    """First `limit` rows taken round-robin over (label, generator), in a fixed
    hash order inside each group, so a trial run covers reals AND every generator."""
    import hashlib
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["generator"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: hashlib.md5(r["video_id"].encode()).hexdigest())
    out, i, keys = [], 0, sorted(groups)
    while len(out) < limit and any(i < len(groups[k]) for k in keys):
        out += [groups[k][i] for k in keys if i < len(groups[k])]
        i += 1
    return out[:limit]


def select_rows(rows, split, limit, done):
    # fakes whose real partner is in the calibration split are never evaluated -> skip
    rows = [r for r in rows if not (r["label"] == "fake" and r["split"] == "calib")]
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    rows = [r for r in rows if r["video_id"] not in done]
    return stratified(rows, limit) if limit else rows


def resume_cleanup(path, cfg, fps_now):
    """Keep only videos that have a completion marker; rewrite the file without
    partial records (atomic replace). Abort if the run configuration differs or a
    completed video_id now points to a different file (e.g. raw vs standardised)."""
    lines, done, old_cfg = [], {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line after a crash
            if r.get("_run_config"):
                old_cfg = r
                continue
            lines.append(r)
            if r.get("_done"):
                done[r["video_id"]] = r.get("src")
    if old_cfg is not None:
        diff = {k: (old_cfg.get(k), v) for k, v in cfg.items() if old_cfg.get(k) != v}
        if diff:
            raise SystemExit(f"--out exists with a different configuration {diff}; use a new --out file")
    clash = [v for v, src in done.items() if v in fps_now and src != fps_now[v]]
    if clash:
        raise SystemExit(f"--out holds results for {len(clash)} video_ids that now point to different "
                         f"files (e.g. {clash[0]}: {done[clash[0]]} -> {fps_now[clash[0]]}); "
                         f"use a new --out file for this manifest")
    kept = [r for r in lines if r["video_id"] in done]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg) + "\n")
        for r in kept:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    return set(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vjepa2", "videomae"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", default=None,
                    help="comma list of mask specs; default vjepa2: future,block:0.25 ; videomae: tube:0.9,future")
    ap.add_argument("--seeds", type=int, default=3, help="seeds for random masks (tube/block)")
    ap.add_argument("--clip-fps", type=float, default=8.0, help="frames are sampled by time at this rate")
    ap.add_argument("--max-clips", type=int, default=4)
    ap.add_argument("--skip-start", type=float, default=0.5,
                    help="seconds skipped at the start (I2V fakes share the real first frame)")
    ap.add_argument("--crops", choices=["center", "three"], default="three")
    ap.add_argument("--split", default="all", choices=["all", "calib", "test"])
    ap.add_argument("--limit", type=int, default=0,
                    help="trial run: N videos sampled evenly over reals and every generator")
    ap.add_argument("--model-id", default=None, help="videomae HF id")
    ap.add_argument("--hub-name", default=None, help="vjepa2 torch.hub entry")
    ap.add_argument("--hub-repo", default=None,
                    help="vjepa2 code; pin it: facebookresearch/vjepa2:<commit-sha>")
    ap.add_argument("--weights", default=None,
                    help="vjepa2 checkpoint URL or local .pt (default: official dl.fbaipublicfiles.com)")
    ap.add_argument("--size", type=int, default=None, help="vjepa2 input size (default 256)")
    ap.add_argument("--num-frames", type=int, default=None, help="vjepa2 frames (default 16)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--prefetch", type=int, default=6, help="videos decoded ahead on CPU threads")
    ap.add_argument("--decode-workers", type=int, default=6)
    a = ap.parse_args()

    masks = (a.masks or ("future,block:0.25" if a.backend == "vjepa2" else "tube:0.9,future")).split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    be = build_backend(a.backend, model_id=a.model_id, hub_name=a.hub_name, hub_repo=a.hub_repo,
                       weights=a.weights, size=a.size, num_frames=a.num_frames, device=dev,
                       amp=not a.no_amp)
    print(f"backend={a.backend} grid={be.grid} size={be.size} frames={be.num_frames} "
          f"decoder={decoder_backend()} device={dev} masks={masks}", flush=True)

    # run configuration: everything that changes the numbers must match on resume
    cfg = {"_run_config": True, "backend": a.backend, "masks": masks, "seeds": a.seeds,
           "clip_fps": a.clip_fps, "max_clips": a.max_clips, "skip_start": a.skip_start,
           "crops": a.crops, "model_id": a.model_id, "hub_name": a.hub_name, "hub_repo": a.hub_repo,
           "weights": a.weights, "size": be.size, "num_frames": be.num_frames, "amp": not a.no_amp}
    all_rows = read_manifest(a.manifest)
    fps_now = {r["video_id"]: fingerprint(r) for r in all_rows}
    done = set()
    if os.path.exists(a.out):
        done = resume_cleanup(a.out, cfg, fps_now)
        print(f"resume: {len(done)} videos complete; incomplete ones are recomputed", flush=True)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(cfg) + "\n")
    rows = select_rows(all_rows, a.split, a.limit, done)
    print(f"{len(rows)} videos to process", flush=True)

    # fixed masks (same for every video -> comparable across videos)
    mask_bank = []
    for spec in masks:
        for seed in (range(a.seeds) if is_random(spec) else [0]):
            mask_bank.append((spec, seed, make_mask(spec, be.grid, seed)))
    T_slots = be.grid[0]

    def prepare(r):
        try:
            return _prepare(r)
        except Exception as e:  # decode failure -> keep the row so it can be marked
            return r, None, f"{type(e).__name__}: {e}"

    def _prepare(r):
        """CPU side: plan clips, decode, preprocess."""
        dur = float(r["duration"]) if r.get("duration") else None
        if not dur:
            return r, None, "no duration"
        clips = plan_clips(dur, a.clip_fps, be.num_frames, a.skip_start, a.max_clips)
        if not clips:
            return r, None, f"too short ({dur:.2f}s)"
        flat = [t for c in clips for t in c]
        fps = float(r["fps"]) if r.get("fps") else None
        frames = decode_frames_at(r["path"], flat, fps)
        out = []
        for ci, c in enumerate(clips):
            fr = frames[ci * be.num_frames:(ci + 1) * be.num_frames]
            x, raw = frames_to_inputs(fr, be.size, a.crops, be.mean, be.std)
            out.append((ci, c[0], x, covariates(raw)))
        return r, out, None

    t0, n_ok, n_err = time.time(), 0, 0
    with ThreadPoolExecutor(a.decode_workers) as ex, open(a.out, "a", encoding="utf-8") as fout:
        it = iter(rows)
        q = deque()
        for r in it:
            q.append(ex.submit(prepare, r))
            if len(q) >= a.prefetch:
                break
        while q:
            fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append(ex.submit(prepare, nxt))
            r, clips, err = fut.result()
            base = {k: r.get(k, "") for k in ("video_id", "label", "generator", "dataset", "source",
                                               "split", "pair_id")}
            src = fingerprint(r)

            def mark_error(msg):
                fout.write(json.dumps({**base, "error": msg}) + "\n")
                fout.write(json.dumps({"video_id": base["video_id"], "_done": True, "n": 0, "src": src}) + "\n")
                fout.flush()

            if err or clips is None:
                n_err += 1
                mark_error(err)
                print(f"[skip] {base['video_id']}: {err}", flush=True)
                continue
            buf = []
            try:
                for ci, tstart, x, cov in clips:
                    for spec, seed, m in mask_bank:
                        e = be.token_errors(x, m)                     # (n_crops, N)
                        e = e.view(e.shape[0], T_slots, -1)           # (n_crops, T', H'W')
                        tgt = e[~torch.isnan(e)]
                        tcurve = []
                        for s in range(T_slots):
                            v = e[:, s][~torch.isnan(e[:, s])]
                            tcurve.append(round(v.mean().item(), 6) if v.numel() else None)
                        per_crop = [round(torch.nanmean(e[k]).item(), 6) for k in range(e.shape[0])]
                        rec = {**base, "clip": ci, "t_start": round(tstart, 3), "mask": spec, "seed": seed,
                               "err_mean": round(tgt.mean().item(), 6),
                               "err_median": round(tgt.median().item(), 6),
                               "err_p90": round(torch.quantile(tgt, 0.9).item(), 6),
                               "err_max": round(tgt.max().item(), 6),
                               "err_std": round(tgt.std().item(), 6),
                               "err_per_crop": per_crop,
                               "temporal_curve": tcurve,
                               "motion": round(cov["motion"], 6), "texture": round(cov["texture"], 6),
                               "backend": a.backend, "error_kind": be.error_kind}
                        buf.append(json.dumps(rec))
                # all records of a video + its completion marker in a single write
                buf.append(json.dumps({"video_id": base["video_id"], "_done": True, "n": len(buf), "src": src}))
                fout.write("\n".join(buf) + "\n")
                fout.flush()
                n_ok += 1
            except Exception as e:  # noqa
                n_err += 1
                mark_error(f"{type(e).__name__}: {e}")
                traceback.print_exc()
            if (n_ok + n_err) % 25 == 0:
                el = time.time() - t0
                print(f"{n_ok + n_err}/{len(rows)} ok={n_ok} err={n_err} {el / (n_ok + n_err):.2f}s/video "
                      f"peak_mem={torch.cuda.max_memory_allocated() / 2**30 if dev == 'cuda' else 0:.1f}GB",
                      flush=True)
    print(f"finished ok={n_ok} err={n_err} in {(time.time() - t0) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
