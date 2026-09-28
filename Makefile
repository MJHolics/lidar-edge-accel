# WSL2 Ubuntu 24.04 · CUDA 12.8 · RTX 4080 SUPER(sm_89)
NVCC  ?= /usr/local/cuda-12.8/bin/nvcc
ARCH  ?= sm_89
# FMAD=false로 두면 GPU도 CPU처럼 곱셈·덧셈을 따로 반올림한다
FMAD  ?= true
BUILD ?= $(HOME)/lidar-build
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17 $(EXTRA)

all: $(BUILD)/bench

$(BUILD)/cpu_ops.o: src/cpu_ops.cpp src/cpu_ops.hpp src/common.hpp
	mkdir -p $(BUILD) && g++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/gpu_ops.o: src/gpu_ops.cu src/gpu_ops.hpp src/common.hpp
	mkdir -p $(BUILD) && $(NVCC) -O3 -arch=$(ARCH) -std=c++17 --fmad=$(FMAD) -c $< -o $@

$(BUILD)/bench: src/bench.cpp $(BUILD)/cpu_ops.o $(BUILD)/gpu_ops.o
	$(NVCC) -O3 -arch=$(ARCH) -std=c++17 -Xcompiler "$(CXXFLAGS)" $^ -o $@ -lgomp

clean:
	rm -rf $(BUILD)
