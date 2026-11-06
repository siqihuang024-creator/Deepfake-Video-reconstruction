#!/usr/bin/env python3
"""Make a format-standardised copy of every video in a manifest (originals are
kept untouched) and write a new manifest with the same video_ids.

Default: fps=24, height=720 (aspect kept, even width), H.264 CRF 23, yuv420p, no audio.
This reduces -- but cannot remove -- format differences (prior compression,
sharpening, frame interpolation stay baked in).

    python scripts/standardize.py --manifest manifests/magic_raw.csv \
        --out-root /data/checkpoints/shuang/std_videos --out manifests/magic_std.csv
"""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe.video_io import ffprobe_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out", required=True, help="output manifest csv")
    ap.add_argument("--fps", type=float, default=24)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads-per-job", type=int, default=2)
    a = ap.parse_args()

    with open(a.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())

    def run(r):
        if r.get("probe_error"):
            return r
        dst = os.path.join(a.out_root, r["video_id"] + ".mp4")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            tmp = dst + ".part.mp4"
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", r["path"], "-an",
                   "-vf", f"fps={a.fps},scale=-2:{a.height}:flags=bicubic",
                   "-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf),
                   "-pix_fmt", "yuv420p", "-threads", str(a.threads_per_job), tmp]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                r = dict(r)
                r["probe_error"] = "ffmpeg: " + p.stderr.strip()[:200]
                return r
            os.replace(tmp, dst)
        r = dict(r)
        r["path"] = os.path.abspath(dst)
        try:
            r.update(ffprobe_meta(dst))
            r["probe_error"] = ""
        except Exception as e:  # noqa
            r["probe_error"] = str(e)[:200]
        return r

    out = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(run, rows), 1):
            out.append(r)
            if i % 100 == 0:
                print(f"{i}/{len(rows)}", flush=True)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    nerr = sum(1 for r in out if r.get("probe_error"))
    print(f"done: {len(out)} rows, {nerr} errors -> {a.out}")


if __name__ == "__main__":
    main()
