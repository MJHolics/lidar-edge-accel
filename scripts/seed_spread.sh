#!/usr/bin/env bash
# 같은 스캔에서 RANSAC 시드만 바꿨을 때 지면·클러스터 수의 흔들림. 사용(WSL): bash scripts/seed_spread.sh
set -e
cd "$(dirname "$0")/.."
B="$HOME/lidar-build-web"
g++ -O3 -std=c++17 -march=native src/web_tool.cpp src/web_api.cpp src/cpu_ops.cpp -o "$B/web_tool_native"
"$B/web_tool_native" seeds web/data 12 50 > results/web_seed_spread.csv
wc -l results/web_seed_spread.csv
