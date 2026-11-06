#!/usr/bin/env python3
"""Aggregate probe results per video, calibrate on REAL calib videos only, and
report per-generator AUROC (+ bootstrap 95% CI) for several score directions.

Score types (all oriented so that larger = "more anomalous / predicted fake"):
    high        raw error                       (fake = harder to predict)
    low         -raw error                      (fake = too easy to predict)
    two         two-tailed ECDF rank of log error vs. calib-real distribution
    r_high/r_low/r_two   same, after regressing log error on log(motion), log(texture)
                         using calib-real videos only (scene-difficulty calibration)

Confound checks (same AUROC protocol, reported as separability = max(AUC, 1-AUC)):
    format: bpp, fps, height, duration (from manifest)
    content: motion, texture covariates
If a format feature alone separates real/fake, the probe score may be riding on it.

Paired metric (when the manifest has pair_id, e.g. CoCoVideo / Magic Videos):
    paired win-rate = P(score(fake) > score(its own real source)), over test pairs.
    This removes content differences between generators' prompt sets; compare it
    with the unpaired AUC -- a big gap means the unpaired AUC is partly content.

Only videos with a completion marker (written by run_probe.py) are used; fakes
whose real partner is in the calib split are excluded.

    python scripts/analyze.py --results results/vjepa2_magic_raw.jsonl \
        --manifest manifests/magic_raw.csv --out analysis/vjepa2_magic_raw
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def load_results(path):
    recs, done, n_err = [], set(), 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("_run_config"):
                print(f"run config: {r}")
            elif r.get("_done"):
                done.add(r["video_id"])
            elif "error" in r:
                n_err += 1
            else:
                recs.append(r)
    kept = [r for r in recs if r["video_id"] in done]
    print(f"results: {len({r['video_id'] for r in kept})} complete videos, "
          f"{len({r['video_id'] for r in recs}) - len({r['video_id'] for r in kept})} incomplete (ignored), "
          f"{n_err} errors")
    return kept


def per_video(recs, man):
    """-> {mask: {video_id: dict(score, score_max, motion, texture, curve, meta...)}}"""
    acc = defaultdict(lambda: defaultdict(lambda: {"clip": defaultdict(list), "motion": [], "texture": [],
                                                   "curves": []}))
    meta = {}
    for r in recs:
        a = acc[r["mask"]][r["video_id"]]
        a["clip"][r["clip"]].append(r["err_mean"])
        a["motion"].append(r["motion"])
        a["texture"].append(r["texture"])
        a["curves"].append([np.nan if v is None else v for v in r["temporal_curve"]])
        m = man.get(r["video_id"], {})
        meta[r["video_id"]] = {"vid": r["video_id"], "pair_id": r.get("pair_id") or m.get("pair_id", ""),
                               # split from the manifest wins (pair-level split may postdate the run)
                               "split": m.get("split") or r["split"],
                               **{k: r[k] for k in ("label", "generator", "dataset")}}
    out = {}
    for mask, vids in acc.items():
        out[mask] = {}
        for vid, a in vids.items():
            clip_means = [float(np.mean(v)) for v in a["clip"].values()]
            out[mask][vid] = dict(meta[vid], score=float(np.mean(clip_means)), score_max=float(np.max(clip_means)),
                                  motion=float(np.mean(a["motion"])), texture=float(np.mean(a["texture"])),
                                  curve=np.nanmean(np.array(a["curves"], dtype=float), 0))
    return out


def ecdf_two_tailed(ref, x):
    ref = np.sort(ref)
    n = len(ref)
    F = (np.searchsorted(ref, x, side="right") + 0.5) / (n + 1)
    return 1 - 2 * np.minimum(F, 1 - F)  # 0 = typical, ->1 = extreme on either side


def fit_resid(calib_y, calib_X, y, X):
    A = np.c_[np.ones(len(calib_X)), calib_X]
    coef, *_ = np.linalg.lstsq(A, calib_y, rcond=None)
    return y - np.c_[np.ones(len(X)), X] @ coef, calib_y - A @ coef, coef


def auc_ci(y_real, y_fake, n_boot=1000, seed=0):
    y = np.r_[np.zeros(len(y_real)), np.ones(len(y_fake))]
    s = np.r_[y_real, y_fake]
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        a = rng.choice(y_real, len(y_real))
        b = rng.choice(y_fake, len(y_fake))
        bs.append(roc_auc_score(y, np.r_[a, b]))
    return auc, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_win(s_real, s_fake, n_boot=1000, seed=0):
    d = np.sign(np.asarray(s_fake) - np.asarray(s_real))
    w = (d > 0) + 0.5 * (d == 0)
    rng = np.random.default_rng(seed)
    bs = [w[rng.integers(0, len(w), len(w))].mean() for _ in range(n_boot)]
    return float(w.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def safe_log(v):
    return np.log(np.maximum(np.asarray(v, dtype=float), 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--agg", choices=["score", "score_max"], default="score",
                    help="video score = mean over clips, or max over clips")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.manifest, encoding="utf-8") as f:
        man = {r["video_id"]: r for r in csv.DictReader(f)}
    pv = per_video(load_results(a.results), man)

    rows, conf_rows = [], []
    for mask, vids in sorted(pv.items()):
        for ds in sorted({v["dataset"] for v in vids.values()}):
            V = {k: v for k, v in vids.items() if v["dataset"] == ds}
            calib = [v for v in V.values() if v["label"] == "real" and v["split"] == "calib"]
            real_test = [v for v in V.values() if v["label"] == "real" and v["split"] == "test"]
            fakes = [v for v in V.values() if v["label"] == "fake" and v["split"] == "test"]
            gens = sorted({v["generator"] for v in fakes})
            if len(calib) < 20 or len(real_test) < 20:
                print(f"[warn] {ds}/{mask}: calib={len(calib)} real_test={len(real_test)} (too few)")
                if not calib or not real_test:
                    continue
            cy = safe_log([v[a.agg] for v in calib])
            cX = np.c_[safe_log([v["motion"] for v in calib]), safe_log([v["texture"] for v in calib])]

            def scores(vs):
                y = safe_log([v[a.agg] for v in vs])
                X = np.c_[safe_log([v["motion"] for v in vs]), safe_log([v["texture"] for v in vs])]
                res, cres, _ = fit_resid(cy, cX, y, X)
                return {"high": y, "low": -y, "two": ecdf_two_tailed(cy, y),
                        "r_high": res, "r_low": -res, "r_two": ecdf_two_tailed(cres, res)}

            s_real = scores(real_test)
            real_pos = {v["pair_id"]: i for i, v in enumerate(real_test) if v["pair_id"]}
            groups = [(g, [v for v in fakes if v["generator"] == g]) for g in gens]
            groups.append(("ALL_FAKES", fakes))
            for g, fk in groups:
                if not fk:
                    continue
                s_fake = scores(fk)
                pairs = [(real_pos[v["pair_id"]], j) for j, v in enumerate(fk) if v["pair_id"] in real_pos]
                for st in s_real:
                    auc, lo, hi = auc_ci(s_real[st], s_fake[st], a.n_boot)
                    row = dict(dataset=ds, mask=mask, score=st, generator=g, n_real=len(real_test),
                               n_fake=len(fk), auc=round(auc, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                               med_real=round(float(np.median([v[a.agg] for v in real_test])), 6),
                               med_fake=round(float(np.median([v[a.agg] for v in fk])), 6),
                               n_pairs=len(pairs), pwin="", pwin_lo="", pwin_hi="")
                    if len(pairs) >= 10:
                        ri, fj = map(list, zip(*pairs))
                        pw, plo, phi = paired_win(s_real[st][ri], s_fake[st][fj], a.n_boot)
                        row.update(pwin=round(pw, 4), pwin_lo=round(plo, 4), pwin_hi=round(phi, 4))
                    rows.append(row)
                # confounds (only once per dataset -> first mask)
                if mask == sorted(pv)[0]:
                    feats = {"motion": lambda v, m: v["motion"], "texture": lambda v, m: v["texture"]}
                    for key in ("bpp", "fps", "height", "duration"):
                        feats[key] = (lambda key: (lambda v, m: float(m[key]) if m.get(key) not in (None, "") else np.nan))(key)
                    for fn, fget in feats.items():
                        xr = np.array([fget(v, man.get(v["vid"], {})) for v in real_test], float)
                        xf = np.array([fget(v, man.get(v["vid"], {})) for v in fk], float)
                        xr, xf = xr[~np.isnan(xr)], xf[~np.isnan(xf)]
                        if len(xr) < 5 or len(xf) < 5 or np.ptp(np.r_[xr, xf]) == 0:
                            sep = 0.5  # missing or constant feature -> carries no information
                        else:
                            auc = roc_auc_score(np.r_[np.zeros(len(xr)), np.ones(len(xf))], np.r_[xr, xf])
                            sep = max(auc, 1 - auc)
                        conf_rows.append(dict(dataset=ds, generator=g, feature=fn, separability=round(sep, 4)))

    def write_csv(path, rs):
        if not rs:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)

    write_csv(os.path.join(a.out, "auc.csv"), rows)
    write_csv(os.path.join(a.out, "confounds.csv"), conf_rows)

    # markdown summary
    lines = [f"# Probe analysis: `{os.path.basename(a.results)}`  (video score = {a.agg})", ""]
    for ds in sorted({r["dataset"] for r in rows}):
        for mask in sorted({r["mask"] for r in rows if r["dataset"] == ds}):
            sub = [r for r in rows if r["dataset"] == ds and r["mask"] == mask]
            gens = list(dict.fromkeys(r["generator"] for r in sub))
            sts = list(dict.fromkeys(r["score"] for r in sub))
            lines += [f"## {ds} | mask `{mask}`", "", "| generator | n_fake | med real | med fake | " +
                      " | ".join(sts) + " |", "|" + "---|" * (4 + len(sts))]
            for g in gens:
                rg = {r["score"]: r for r in sub if r["generator"] == g}
                any_r = next(iter(rg.values()))
                cells = [f"{rg[s]['auc']:.3f} [{rg[s]['ci_lo']:.2f},{rg[s]['ci_hi']:.2f}]" for s in sts]
                lines.append(f"| {g} | {any_r['n_fake']} | {any_r['med_real']:.4g} | {any_r['med_fake']:.4g} | " +
                             " | ".join(cells) + " |")
            lines.append("")
            psub = [r for r in sub if r["pwin"] != ""]
            if psub:
                lines += [f"**paired win-rate** P(fake > its own real), same score types:", "",
                          "| generator | n_pairs | " + " | ".join(sts) + " |", "|" + "---|" * (2 + len(sts))]
                for g in gens:
                    rg = {r["score"]: r for r in psub if r["generator"] == g}
                    if not rg:
                        continue
                    cells = [f"{rg[s]['pwin']:.3f} [{rg[s]['pwin_lo']:.2f},{rg[s]['pwin_hi']:.2f}]" if s in rg else "-"
                             for s in sts]
                    lines.append(f"| {g} | {next(iter(rg.values()))['n_pairs']} | " + " | ".join(cells) + " |")
                lines.append("")
        cr = [r for r in conf_rows if r["dataset"] == ds]
        if cr:
            feats = list(dict.fromkeys(r["feature"] for r in cr))
            lines += [f"## {ds} | confound separability (max(AUC,1-AUC); >0.8 = warning)", "",
                      "| generator | " + " | ".join(feats) + " |", "|" + "---|" * (1 + len(feats))]
            for g in dict.fromkeys(r["generator"] for r in cr):
                d = {r["feature"]: r["separability"] for r in cr if r["generator"] == g}
                lines.append(f"| {g} | " + " | ".join(
                    (f"**{d[x]:.2f}**" if d.get(x, 0) > 0.8 else f"{d.get(x, float('nan')):.2f}") for x in feats) + " |")
            lines.append("")
    lines += ["Notes: AUC 0.5 = no signal. `high` > 0.5 means fakes are harder to predict; `low` > 0.5 means "
              "fakes are easier. Thresholds/normal ranges use calib-real only; fakes are used only for evaluation."]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    # plots (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed -> skipping plots")
        return
    for mask, vids in pv.items():
        for ds in sorted({v["dataset"] for v in vids.values()}):
            V = [v for v in vids.values() if v["dataset"] == ds]
            groups = defaultdict(list)
            for v in V:
                key = "real" if v["label"] == "real" else v["generator"]
                if v["split"] == "calib":
                    continue
                groups[key].append(v)
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
            allv = safe_log([v[a.agg] for v in V])
            bins = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 50)
            for k, vs in sorted(groups.items(), key=lambda kv: kv[0] != "real"):
                ax[0].hist(safe_log([v[a.agg] for v in vs]), bins=bins, histtype="step", density=True,
                           lw=2.2 if k == "real" else 1.2, color="k" if k == "real" else None, label=f"{k} ({len(vs)})")
                curves = np.array([v["curve"] for v in vs])
                mu = np.nanmean(curves, 0)
                ax[1].plot(np.arange(len(mu)), mu, marker="o", lw=2.2 if k == "real" else 1.2,
                           color="k" if k == "real" else None, label=k)
            ax[0].set_xlabel(f"log video error ({a.agg})"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=7)
            ax[1].set_xlabel("temporal token slot"); ax[1].set_ylabel("mean error (target slots)")
            fig.suptitle(f"{ds} | {mask}")
            fig.tight_layout()
            fn = os.path.join(a.out, f"dist_{ds}_{mask.replace(':', '-')}.png")
            fig.savefig(fn, dpi=130)
            plt.close(fig)
    print(f"plots -> {a.out}")


if __name__ == "__main__":
    main()
