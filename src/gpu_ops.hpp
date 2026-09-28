#pragma once
#include "common.hpp"

enum class VoxelMode { ThrustSort, Hash };       // 라이브러리 정렬 vs 직접 쓴 해시 커널
// atomicMin 유니온-파인드 vs 반복 라벨 전파(매 반복 수렴 확인) vs 전파(16회마다 확인)
enum class ClusterMode { UnionFind, Propagate, Propagate16 };

class GpuPipeline {
 public:
  GpuPipeline(const Params& p, int max_points);
  ~GpuPipeline();
  // validate=true면 복셀 목록·라벨 배열까지 내려받는다(시간 측정 밖에서).
  StageMs run(const std::vector<P4>& pts, uint32_t seed, VoxelMode vm, ClusterMode cm,
              bool validate, Result& r);
  const char* device_name() const;

 private:
  struct Impl;
  Impl* d;
};
