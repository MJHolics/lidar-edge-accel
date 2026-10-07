#!/usr/bin/env bash
# WSL2 Ubuntu에 Emscripten(em++) 설치. 같은 C++ 소스를 브라우저용 WASM으로 내는 데 쓴다.
set -e
cd "$HOME"
[ -d emsdk ] || git clone -q --depth 1 https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 4.0.15
./emsdk activate 4.0.15
source ./emsdk_env.sh
em++ --version | head -1
