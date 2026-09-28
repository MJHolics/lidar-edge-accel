// 공통 정의: CPU(g++)와 GPU(nvcc) 양쪽에서 같은 식을 쓰도록 키·평면 계산을 여기 둔다.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

struct P4 { float x, y, z, i; };          // KITTI velodyne .bin 한 점(16바이트)
struct Voxel { uint32_t key; float x, y, z; };  // 복셀 키 + 중심(평균)
struct Plane { float a, b, c, d; };

struct Params {
  float xmin = -40.f, xmax = 40.f, ymin = -40.f, ymax = 40.f, zmin = -3.f, zmax = 2.f;
  float voxel = 0.2f;
  int   nx = 400, ny = 400, nz = 25;       // (max-min)/voxel
  int   ransac_iters = 256;
  float ransac_thresh = 0.2f;
  float min_nz = 0.9f;                    // 지면 후보는 거의 수평인 평면만
  float cell = 0.4f;                      // BEV 클러스터 격자
  int   gw = 200, gh = 200;
  int   min_cells = 3;                    // 이보다 작은 성분은 노이즈로 본다
};

static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;

HD inline uint32_t voxel_key(float x, float y, float z, const Params& p) {
  if (x < p.xmin || x >= p.xmax || y < p.ymin || y >= p.ymax || z < p.zmin || z >= p.zmax)
    return EMPTY_KEY;
  int ix = (int)floorf((x - p.xmin) / p.voxel);
  int iy = (int)floorf((y - p.ymin) / p.voxel);
  int iz = (int)floorf((z - p.zmin) / p.voxel);
  if (ix >= p.nx) ix = p.nx - 1;
  if (iy >= p.ny) iy = p.ny - 1;
  if (iz >= p.nz) iz = p.nz - 1;
  return ((uint32_t)iz * p.ny + iy) * p.nx + ix;
}

HD inline bool plane3(float x1, float y1, float z1, float x2, float y2, float z2,
                      float x3, float y3, float z3, float min_nz, Plane& pl) {
  float ux = x2 - x1, uy = y2 - y1, uz = z2 - z1;
  float vx = x3 - x1, vy = y3 - y1, vz = z3 - z1;
  float a = uy * vz - uz * vy, b = uz * vx - ux * vz, c = ux * vy - uy * vx;
  float n = sqrtf(a * a + b * b + c * c);
  if (n < 1e-6f) return false;
  a /= n; b /= n; c /= n;
  if (fabsf(c) < min_nz) return false;
  pl.a = a; pl.b = b; pl.c = c; pl.d = -(a * x1 + b * y1 + c * z1);
  return true;
}

HD inline bool is_inlier(const Plane& pl, float x, float y, float z, float t) {
  return fabsf(pl.a * x + pl.b * y + pl.c * z + pl.d) < t;
}

HD inline int bev_cell(float x, float y, const Params& p) {
  int cx = (int)floorf((x - p.xmin) / p.cell);
  int cy = (int)floorf((y - p.ymin) / p.cell);
  if (cx < 0 || cy < 0 || cx >= p.gw || cy >= p.gh) return -1;
  return cy * p.gw + cx;
}

inline std::vector<P4> load_bin(const std::string& path) {
  std::vector<P4> v;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return v;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  v.resize(sz / sizeof(P4));
  size_t got = std::fread(v.data(), sizeof(P4), v.size(), f);
  v.resize(got);
  std::fclose(f);
  return v;
}

inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// RANSAC 가설(점 인덱스 3개씩)을 모든 구현이 공유하도록 호스트에서 한 번 만든다.
// std::mt19937 대신 xorshift를 써서 컴파일러·표준 라이브러리와 무관하게 재현되게 했다.
inline std::vector<int> make_hypotheses(int n, int iters, uint32_t seed) {
  std::vector<int> h(3 * iters);
  uint32_t s = seed ? seed : 1u;
  for (auto& x : h) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    x = n > 0 ? (int)(s % (uint32_t)n) : 0;
  }
  return h;
}

// 단계별 결과 요약 — 구현 간 일치 확인에 쓴다.
struct Result {
  std::vector<Voxel> voxels;     // key 오름차순
  int best_hyp = -1, best_score = -1;
  int n_ground = 0;
  std::vector<int> labels;       // BEV 셀별 성분 대표(성분 내 최소 셀 인덱스), 빈 셀은 -1
  int n_clusters = 0;            // min_cells 이상 성분 수
  int prop_iters = 0;            // 라벨 전파 방식일 때 반복 횟수
};

struct StageMs { double h2d = 0, voxel = 0, ransac = 0, cluster = 0, d2h = 0, total = 0; };
