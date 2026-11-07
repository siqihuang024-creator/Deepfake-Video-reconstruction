#!/usr/bin/env python3
"""Scan dataset folders and write a manifest CSV with labels and ffprobe metadata.

Each --spec describes one folder (comma-separated key=value):
    root=PATH        folder to scan (required)
    label=real|fake  (required)
    generator=NAME   e.g. real, Wan2.1, Hailuo ... (required)
    dataset=NAME     e.g. magic, cocovideo (required)
    source=NAME      optional, e.g. pexels / mixkit
    glob=PATTERN     optional, default **/*.mp4
    split=auto|calib|test   optional, default auto
         auto : real videos are split calib/test by a hash of the id (--calib-frac),
                fakes always go to test
         calib: whole folder used only for threshold / normal-range calibration
         test : whole folder used only for evaluation
    pair=none|stem|regex:PATTERN|csv:PATH   optional, default none
         links each fake to the real video it was generated from (CoCoVideo: same
         caption + real first frame; Magic Videos: same caption).
         stem         : pair key = file name without extension
         regex:PAT    : pair key = first capture group of PAT on the relative path
                        (no commas inside PAT)
         csv:PATH     : CSV with columns key,pair_id ; key = file stem
         With pairs, the calib/test split is drawn per PAIR, so a real video and all
         its fakes always land on the same side, and analyze.py adds paired metrics.

Example
    python scripts/build_manifest.py --out manifests/magic_raw.csv \
      --spec root=/data/datasets/shuang/magic/real/real_videos_pexels_720p,label=real,generator=real,dataset=magic,source=pexels,pair=stem \
      --spec root=/data/datasets/shuang/magic/fake/pexels_wan2.1-T2V-1.3B,label=fake,generator=pexels_wan2.1-T2V-1.3B,dataset=magic,pair=stem

    The pair=stem example is only valid if the corresponding file stems really match.
    Inspect sample file names and verify pairs before running on the full dataset.
"""
import argparse
import csv
import glob
import hashlib
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe.video_io import ffprobe_meta  # noqa: E402

FIELDS = ["video_id", "path", "label", "generator", "dataset", "source", "pair_id", "split",
          "codec", "width", "height", "fps", "nb_frames", "duration", "bitrate", "bpp",
          "pix_fmt", "size_bytes", "probe_error"]


def parse_spec(s):
    d = dict(kv.split("=", 1) for kv in s.split(",") if kv)
    for k in ("root", "label", "generator", "dataset"):
        if k not in d:
            raise SystemExit(f"--spec missing '{k}': {s}")
    if d["label"] not in ("real", "fake"):
        raise SystemExit(f"label must be real|fake: {s}")
    d.setdefault("glob", "**/*.mp4")
    d.setdefault("split", "auto")
    d.setdefault("source", "")
    d.setdefault("pair", "none")
    return d


_pair_csv_cache = {}


def pair_key(spec, rel):
    import re
    mode = spec["pair"]
    stem = os.path.basename(rel)
    if mode == "none":
        return ""
    if mode == "stem":
        return stem
    if mode.startswith("regex:"):
        m = re.search(mode[6:], rel)
        return m.group(1) if m else ""
    if mode.startswith("csv:"):
        path = mode[4:]
        if path not in _pair_csv_cache:
            with open(path, encoding="utf-8") as f:
                _pair_csv_cache[path] = {r["key"]: r["pair_id"] for r in csv.DictReader(f)}
        return _pair_csv_cache[path].get(stem, "")
    raise SystemExit(f"unknown pair mode {mode}")


def hash_frac(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-pair-rate", type=float, default=0.9,
                    help="abort if a paired generator has fewer fakes with a real partner than this")
    a = ap.parse_args()

    rows = []
    for s in map(parse_spec, a.spec):
        files = sorted(glob.glob(os.path.join(s["root"], s["glob"]), recursive=True))
        if not files:
            print(f"[warn] no files for {s['root']}/{s['glob']}")
        for f in files:
            rel = os.path.splitext(os.path.relpath(f, s["root"]))[0].replace(os.sep, "/")
            vid = f"{s['dataset']}/{s['generator']}/{s['source'] + '/' if s['source'] else ''}{rel}"
            pk = pair_key(s, rel)
            rows.append(dict(video_id=vid, path=os.path.abspath(f), label=s["label"],
                             generator=s["generator"], dataset=s["dataset"], source=s["source"],
                             pair_id=f"{s['dataset']}/{pk}" if pk else "",
                             split_req=s["split"], pair_mode=s["pair"]))

    ids = [r["video_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate video_id -- give folders distinct dataset/generator/source")

    # ---- validate pairing BEFORE assigning splits ----
    real_count = Counter(r["pair_id"] for r in rows if r["label"] == "real" and r["pair_id"])
    dup = [p for p, n in real_count.items() if n > 1]
    if dup:
        raise SystemExit(f"{len(dup)} pair_ids match more than one real video (e.g. {dup[:3]}); "
                         f"the pair rule is ambiguous -- fix pair=... before running")
    paired_any = any(r["pair_mode"] != "none" for r in rows)
    bad = []
    for g in sorted({r["generator"] for r in rows if r["label"] == "fake" and r["pair_mode"] != "none"}):
        fk = [r for r in rows if r["generator"] == g and r["label"] == "fake"]
        rate = sum(r["pair_id"] in real_count for r in fk) / len(fk)
        if rate < a.min_pair_rate:
            bad.append(f"{g}: {rate:.0%}")
    if bad:
        raise SystemExit(f"pairing rate below --min-pair-rate={a.min_pair_rate:.0%} for {bad}; check the "
                         f"pair rule on a few file names (or lower --min-pair-rate deliberately)")
    # a fake keeps its pair_id only if the real partner exists; otherwise it is unpaired
    for r in rows:
        if r["label"] == "fake" and r["pair_id"] not in real_count:
            r["pair_id"] = ""

    for r in rows:
        if r["split_req"] != "auto":
            r["split"] = r["split_req"]
        elif r["pair_id"]:  # split per pair: real and its fakes on the same side
            r["split"] = "calib" if hash_frac(r["pair_id"]) < a.calib_frac else "test"
        elif r["label"] == "real":
            r["split"] = "calib" if hash_frac(r["video_id"]) < a.calib_frac else "test"
        else:  # unpaired fake -> always evaluated
            r["split"] = "test"
    if paired_any:
        # a real forced into calib/test by split= must take its fakes along
        side = {r["pair_id"]: r["split"] for r in rows if r["label"] == "real" and r["pair_id"]}
        for r in rows:
            if r["label"] == "fake" and r["pair_id"]:
                r["split"] = side[r["pair_id"]]

    def probe(r):
        try:
            r.update(ffprobe_meta(r["path"]))
            r["probe_error"] = ""
        except Exception as e:  # noqa
            r["probe_error"] = str(e)[:200]
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(probe, rows))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary
    c = Counter((r["label"], r["generator"], r["split"]) for r in rows)
    print(f"wrote {len(rows)} rows -> {a.out}")
    for k in sorted(c):
        print(f"  {k[0]:5s} {k[1]:20s} {k[2]:6s} {c[k]}")
    agg = defaultdict(list)
    for r in rows:
        if not r["probe_error"]:
            agg[r["generator"]].append(r)
    print("\nformat summary per generator (check for shortcuts!):")
    for g, rs in sorted(agg.items()):
        res = Counter(f"{r['width']}x{r['height']}" for r in rs).most_common(3)
        fps = Counter(r["fps"] for r in rs).most_common(3)
        codec = Counter(r["codec"] for r in rs).most_common(2)
        print(f"  {g:20s} res={res} fps={fps} codec={codec}")
    # pairing report
    if any(r["pair_id"] for r in rows):
        print("\npairing (fakes whose real partner exists):")
        for g in sorted({r["generator"] for r in rows if r["label"] == "fake"}):
            fk = [r for r in rows if r["generator"] == g and r["label"] == "fake"]
            n = sum(1 for r in fk if r["pair_id"])
            print(f"  {g:20s} {n}/{len(fk)} paired (unpaired fakes: no pair_id, split=test, always evaluated)")
        print("  note: fakes in split=calib (partner used for calibration) are not evaluated")
        print("  examples -- CHECK by eye that these really belong together:")
        for r in [r for r in rows if r["label"] == "fake" and r["pair_id"]][:3]:
            real = next(q for q in rows if q["label"] == "real" and q["pair_id"] == r["pair_id"])
            print(f"    {r['path']}\n      <-> {real['path']}")
    nerr = sum(1 for r in rows if r["probe_error"])
    if nerr:
        print(f"[warn] {nerr} files failed ffprobe (see probe_error column)")


if __name__ == "__main__":
    main()
