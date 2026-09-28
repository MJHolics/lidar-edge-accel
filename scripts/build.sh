#!/usr/bin/env bash
# 사용: bash scripts/build.sh [FMAD=true|false]
set -e
cd "$(dirname "$0")/.."
FMAD="${1:-true}"
BUILD="$HOME/lidar-build-fmad-$FMAD"
make -B BUILD="$BUILD" FMAD="$FMAD" 2>&1
echo "built: $BUILD/bench"
