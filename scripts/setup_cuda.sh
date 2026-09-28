#!/usr/bin/env bash
# WSL2 Ubuntu 24.04에 CUDA 툴킷(nvcc) 설치. 드라이버는 Windows 쪽 것을 쓰므로 툴킷만 받는다.
set -e
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q cuda-toolkit-12-8 libomp-dev
/usr/local/cuda-12.8/bin/nvcc --version
