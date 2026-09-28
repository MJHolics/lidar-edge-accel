#!/usr/bin/env bash
# 라벨 전파 1회 반복의 구성: 커널 시간 vs 동기화 왕복
set -e
cd "$(dirname "$0")/.."
NSYS=$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda-12.8/bin/nsys 2>/dev/null | head -1)
D="$HOME/lidar-data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data"
mkdir -p results/nsys
"$NSYS" profile -o results/nsys/prop -f true -t cuda "$HOME/lidar-build-fmad-true/bench" "$D" results/nsys/run 1 10 >/dev/null
"$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o results/nsys/prop results/nsys/prop.nsys-rep >/dev/null
ls results/nsys
