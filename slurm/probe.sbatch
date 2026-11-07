#!/bin/bash
# Usage (from slurm-datakom):
#   sbatch --export=ALL,BACKEND=vjepa2,MANIFEST=manifests/magic_raw.csv,TAG=magic_raw slurm/probe.sbatch
#   sbatch --export=ALL,BACKEND=videomae,MANIFEST=manifests/magic_std.csv,TAG=magic_std slurm/probe.sbatch
# Optional: MASKS="future,block:0.25"  SEEDS=3  EXTRA="--limit 50"
#SBATCH --job-name=pred_probe
# !! Slurm opens the log files BEFORE this script runs: create the folder once on the
# !! login node first:   mkdir -p ~/logs   (otherwise the job has no log / may fail)
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --output=/home/%u/logs/%x_%j.out
#SBATCH --error=/home/%u/logs/%x_%j.err
set -euo pipefail

PROJ="${PROJ:-/home/${USER}/projects/Deepfake Reconstruction/predictability_probe}"
SIF=${SIF:-/home/containers/pytorch-vision-h200.sif}
OVERLAY=${OVERLAY:-/home/${USER}/overlay.img}          # holds extra pip packages (see README)
CACHE=/data/checkpoints/${USER}/cache                  # model weights, NOT in /home
OUTDIR=/data/checkpoints/${USER}/predictability_probe/results
mkdir -p "$CACHE" "$OUTDIR" /home/${USER}/logs

OV_ARG=""
[ -f "$OVERLAY" ] && OV_ARG="--overlay ${OVERLAY}:ro"

echo "[$(date)] $(hostname) backend=${BACKEND} manifest=${MANIFEST} tag=${TAG}"
apptainer exec --nv ${OV_ARG} \
    -B /data/datasets/${USER}:/data/datasets/${USER}:ro \
    -B /data/checkpoints/${USER}:/data/checkpoints/${USER} \
    -B /home/${USER}:/home/${USER} \
    --env HF_HOME=${CACHE}/hf,TORCH_HOME=${CACHE}/torch,PYTHONUNBUFFERED=1 \
    "$SIF" \
    python3 "${PROJ}/scripts/run_probe.py" \
        --backend "${BACKEND}" \
        --manifest "${PROJ}/${MANIFEST}" \
        --out "${OUTDIR}/${BACKEND}_${TAG}.jsonl" \
        ${MASKS:+--masks "$MASKS"} --seeds "${SEEDS:-3}" \
        --decode-workers 12 ${EXTRA:-}

echo "[$(date)] done -> ${OUTDIR}/${BACKEND}_${TAG}.jsonl"
