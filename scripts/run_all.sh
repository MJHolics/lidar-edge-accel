#!/usr/bin/env bash
# 전체 108스캔 측정 3종: 본 설정 / GPU FMA 끔 / CPU 정렬 스레드 수정 전(FIX_SORT_THREADS=0)
set -e
cd "$(dirname "$0")/.."
make -B BUILD="$HOME/lidar-build-fmad-true" FMAD=true >/dev/null
make -B BUILD="$HOME/lidar-build-fmad-false" FMAD=false >/dev/null
make -B BUILD="$HOME/lidar-build-nofix" FMAD=true EXTRA=-DFIX_SORT_THREADS=0 >/dev/null
D="$HOME/lidar-data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data"
mkdir -p results
"$HOME/lidar-build-fmad-true/bench"  "$D" results/main 5
"$HOME/lidar-build-fmad-false/bench" "$D" results/nofma 5
"$HOME/lidar-build-nofix/bench"      "$D" results/nofix 5
echo ALL_DONE
