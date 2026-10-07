#!/usr/bin/env bash
# 웹 화면 준비: 프레임 줄이기 → 네이티브 대조 정답 → 같은 소스를 WASM으로.
# 사용(WSL): bash scripts/build_web.sh   (먼저 scripts/get_kitti.sh, scripts/setup_emsdk.sh)
set -e
cd "$(dirname "$0")/.."
B="$HOME/lidar-build-web"
D="$HOME/lidar-data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data"
START=0; STEP=9; COUNT=12                      # 108장 중 0.9초 간격 12장
SRC="src/web_api.cpp src/cpu_ops.cpp"
mkdir -p "$B" web/data

# 대조 정답은 곱셈-덧셈 합치기(FMA)를 끈 빌드로 낸다. WASM에는 FMA가 없다.
g++ -O3 -std=c++17 -ffp-contract=off src/web_tool.cpp $SRC -o "$B/web_tool"
g++ -O3 -std=c++17 -march=native     src/web_tool.cpp $SRC -o "$B/web_tool_native"

"$B/web_tool" pack "$D" web/data $START $STEP $COUNT 2> "$B/pack.log"
cat "$B/pack.log"
"$B/web_tool"        ref web/data $START $STEP $COUNT > web/data/expected.json
"$B/web_tool_native" ref web/data $START $STEP $COUNT > "$B/expected_native.json"

source "$HOME/emsdk/emsdk_env.sh" > /dev/null 2>&1
FN="_malloc,_free,_lidar_run,_lidar_set_iters,_lidar_xyz,_lidar_plabel,_lidar_vxyz,_lidar_vlabel,_lidar_stat,_lidar_hash,_lidar_ms,_lidar_plane"
em++ -O3 -std=c++17 $SRC -o web/lidar.mjs \
  -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=web,node -sALLOW_MEMORY_GROWTH \
  -sEXPORTED_FUNCTIONS=$FN -sEXPORTED_RUNTIME_METHODS=HEAPU16,HEAPF32,HEAP32
ls -l web/lidar.mjs web/lidar.wasm web/data | awk '{print $5, $9}'
cp "$B/expected_native.json" "$B/pack.log" results/ 2>/dev/null || true
echo WEB_BUILD_DONE
