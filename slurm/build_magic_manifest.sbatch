#!/bin/bash
# Submit from slurm-datakom after creating ~/logs:
#   sbatch slurm/build_magic_manifest.sbatch
# Uses the Magic Videos directory layout confirmed on Hefest.
# Real and fake clips with the same filename stem share one pair_id.
#SBATCH --job-name=magic_manifest
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0
#SBATCH --output=/home/%u/logs/%x_%j.out
#SBATCH --error=/home/%u/logs/%x_%j.err
set -euo pipefail

PROJ="${PROJ:-/home/${USER}/projects/Deepfake Reconstruction/predictability_probe}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/${USER}/magic}"
SIF="${SIF:-/home/containers/pytorch-vision-h200.sif}"
OVERLAY="${OVERLAY:-/home/${USER}/overlay.img}"
OUT="${PROJ}/manifests/magic_raw.csv"

for subdir in \
    real/real_videos_pexels_720p \
    real/real_videos_mixkit_720p \
    fake/step-video \
    fake/jimeng-S3.0 \
    fake/jimeng-S2.0 \
    fake/wanx2.1 \
    fake/hailuo-T2V-01 \
    fake/pexels_wan2.1-T2V-1.3B; do
    if [ ! -d "${DATA_ROOT}/${subdir}" ]; then
        echo "Missing dataset directory: ${DATA_ROOT}/${subdir}" >&2
        exit 1
    fi
done

overlay_args=()
if [ -f "$OVERLAY" ]; then
    overlay_args=(--overlay "${OVERLAY}:ro")
fi

bind_args=(
    -B "/data/datasets/${USER}:/data/datasets/${USER}:ro"
    -B "/home/${USER}:/home/${USER}"
)

# build_manifest.py records ffprobe errors in the CSV, so catch a missing ffprobe
# before scanning the entire dataset.
apptainer exec "${overlay_args[@]}" "${bind_args[@]}" "$SIF" \
    sh -c 'command -v ffprobe >/dev/null || { echo "ffprobe is missing in the container" >&2; exit 1; }'

mkdir -p "${PROJ}/manifests"
apptainer exec "${overlay_args[@]}" "${bind_args[@]}" "$SIF" \
    python3 "${PROJ}/scripts/build_manifest.py" \
    --out "$OUT" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --spec "root=${DATA_ROOT}/real/real_videos_pexels_720p,label=real,generator=real,dataset=magic,source=pexels,pair=stem" \
    --spec "root=${DATA_ROOT}/real/real_videos_mixkit_720p,label=real,generator=real,dataset=magic,source=mixkit,pair=stem" \
    --spec "root=${DATA_ROOT}/fake/step-video,label=fake,generator=step-video,dataset=magic,pair=stem" \
    --spec "root=${DATA_ROOT}/fake/jimeng-S3.0,label=fake,generator=jimeng-S3.0,dataset=magic,pair=stem" \
    --spec "root=${DATA_ROOT}/fake/jimeng-S2.0,label=fake,generator=jimeng-S2.0,dataset=magic,pair=stem" \
    --spec "root=${DATA_ROOT}/fake/wanx2.1,label=fake,generator=wanx2.1,dataset=magic,pair=stem" \
    --spec "root=${DATA_ROOT}/fake/hailuo-T2V-01,label=fake,generator=hailuo-T2V-01,dataset=magic,pair=stem" \
    --spec "root=${DATA_ROOT}/fake/pexels_wan2.1-T2V-1.3B,label=fake,generator=pexels_wan2.1-T2V-1.3B,dataset=magic,pair=stem"

echo "Manifest written to $OUT"
