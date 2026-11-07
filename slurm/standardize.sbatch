#!/bin/bash
# CPU-only job: make the format-standardised copy of a dataset.
#   sbatch --export=ALL,MANIFEST=manifests/magic_raw.csv,TAG=magic slurm/standardize.sbatch
# Needs ffmpeg/ffprobe (in the container or the overlay -- check with `which ffmpeg`).
#SBATCH --job-name=std_videos
# !! Slurm opens the log files BEFORE this script runs: create the folder once on the
# !! login node first:   mkdir -p ~/logs   (otherwise the job has no log / may fail)
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0
#SBATCH --output=/home/%u/logs/%x_%j.out
#SBATCH --error=/home/%u/logs/%x_%j.err
set -euo pipefail

PROJ="${PROJ:-/home/${USER}/projects/Deepfake Reconstruction/predictability_probe}"
SIF=${SIF:-/home/containers/pytorch-vision-h200.sif}
OVERLAY=${OVERLAY:-/home/${USER}/overlay.img}
# Standardised videos go to /data/checkpoints (writable); /data/datasets is mounted read-only.
STDROOT=${STDROOT:-/data/checkpoints/${USER}/std_videos}
mkdir -p "$STDROOT" /home/${USER}/logs
OV_ARG=""
[ -f "$OVERLAY" ] && OV_ARG="--overlay ${OVERLAY}:ro"

apptainer exec ${OV_ARG} \
    -B /data/datasets/${USER}:/data/datasets/${USER}:ro \
    -B /data/checkpoints/${USER}:/data/checkpoints/${USER} \
    -B /home/${USER}:/home/${USER} \
    "$SIF" \
    python3 "${PROJ}/scripts/standardize.py" \
        --manifest "${PROJ}/${MANIFEST}" \
        --out-root "${STDROOT}" \
        --out "${PROJ}/manifests/${TAG}_std.csv" \
        --workers 8 --threads-per-job 2
