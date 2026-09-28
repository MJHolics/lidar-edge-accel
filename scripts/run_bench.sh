#!/usr/bin/env bash
# 사용: bash scripts/run_bench.sh <tag> [FMAD=true] [reps=5] [max_scans=0]
set -e
cd "$(dirname "$0")/.."
TAG="$1"; FMAD="${2:-true}"; REPS="${3:-5}"; MAX="${4:-0}"
mkdir -p results
"$HOME/lidar-build-fmad-$FMAD/bench" \
  "$HOME/lidar-data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data" \
  "results/$TAG" "$REPS" "$MAX"
