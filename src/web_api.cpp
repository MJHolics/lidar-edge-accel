// 브라우저(WASM)와 네이티브 대조 도구가 함께 쓰는 얇은 층. 계산은 cpu_ops.cpp의 직렬 경로 그대로다.
// 입력은 ROI 안 점을 uint16 3개(x, y, z)로 줄인 것 — 한 점 6바이트(원본 .bin은 16바이트).
#include "web_api.hpp"
#include "cpu_ops.hpp"
#include <algorithm>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

static Params P;
static Result R;
static StageMs T;
static Plane PL{0, 0, 1, 0};
static std::vector<P4> pts;
static std::vector<float> xyz, vxyz;
static std::vector<int32_t> plabel, vlabel;
static int n_objects = 0, n_small = 0;
static uint32_t h_keys = 0, h_labels = 0, h_centroids = 0;

static uint32_t fnv(const void* data, size_t bytes, uint32_t h = 2166136261u) {
  const uint8_t* b = (const uint8_t*)data;
  for (size_t i = 0; i < bytes; ++i) { h ^= b[i]; h *= 16777619u; }
  return h;
}

std::vector<P4> unpack_points(const uint16_t* q, int n, const Params& p) {
  const float sx = (p.xmax - p.xmin) / 65536.f, sy = (p.ymax - p.ymin) / 65536.f, sz = (p.zmax - p.zmin) / 65536.f;
  std::vector<P4> v(n);
  for (int i = 0; i < n; ++i)
    v[i] = {p.xmin + ((float)q[3 * i] + 0.5f) * sx, p.ymin + ((float)q[3 * i + 1] + 0.5f) * sy,
            p.zmin + ((float)q[3 * i + 2] + 0.5f) * sz, 0.f};
  return v;
}

extern "C" {

// 한 스캔을 돌린다. 반환값은 복셀 수.
EMSCRIPTEN_KEEPALIVE int lidar_run(const uint16_t* q, int n, uint32_t seed) {
  pts = unpack_points(q, n, P);
  T = run_cpu(pts, P, 1, seed, R);

  // 아래는 그리기용 정리(시간에 넣지 않는다): 복셀·점마다 지면/물체/작은 조각을 붙인다.
  auto hyp = make_hypotheses((int)R.voxels.size(), P.ransac_iters, seed);
  bool ok = false;
  if (R.best_hyp >= 0) {
    const Voxel &a = R.voxels[hyp[3 * R.best_hyp]], &b = R.voxels[hyp[3 * R.best_hyp + 1]], &c = R.voxels[hyp[3 * R.best_hyp + 2]];
    ok = plane3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, P.min_nz, PL);
  }
  std::vector<int> size(P.gw * P.gh, 0);
  for (int l : R.labels) if (l >= 0) ++size[l];
  n_objects = n_small = 0;
  for (int s : size) { if (s >= P.min_cells) ++n_objects; else if (s > 0) ++n_small; }

  const int nv = (int)R.voxels.size();
  vxyz.resize(3 * nv); vlabel.resize(nv);
  for (int i = 0; i < nv; ++i) {
    const Voxel& v = R.voxels[i];
    vxyz[3 * i] = v.x; vxyz[3 * i + 1] = v.y; vxyz[3 * i + 2] = v.z;
    int lab = -1;                                             // -2 지면, -1 작은 조각, 0 이상 = 물체(성분 대표 셀)
    if (ok && is_inlier(PL, v.x, v.y, v.z, P.ransac_thresh)) lab = -2;
    else { int cell = bev_cell(v.x, v.y, P); if (cell >= 0 && R.labels[cell] >= 0 && size[R.labels[cell]] >= P.min_cells) lab = R.labels[cell]; }
    vlabel[i] = lab;
  }
  xyz.resize(3 * n); plabel.resize(n);
  for (int i = 0; i < n; ++i) {
    xyz[3 * i] = pts[i].x; xyz[3 * i + 1] = pts[i].y; xyz[3 * i + 2] = pts[i].z;
    uint32_t key = voxel_key(pts[i].x, pts[i].y, pts[i].z, P);
    auto it = std::lower_bound(R.voxels.begin(), R.voxels.end(), key, [](const Voxel& v, uint32_t k) { return v.key < k; });
    plabel[i] = (it != R.voxels.end() && it->key == key) ? vlabel[it - R.voxels.begin()] : -1;
  }

  h_keys = h_centroids = 2166136261u;
  for (const Voxel& v : R.voxels) { h_keys = fnv(&v.key, 4, h_keys); h_centroids = fnv(&v.x, 12, h_centroids); }
  h_labels = fnv(R.labels.data(), R.labels.size() * sizeof(int));
  return nv;
}

EMSCRIPTEN_KEEPALIVE void lidar_set_iters(int n) { P.ransac_iters = n; }   // 기본 256
EMSCRIPTEN_KEEPALIVE float* lidar_xyz() { return xyz.data(); }
EMSCRIPTEN_KEEPALIVE int32_t* lidar_plabel() { return plabel.data(); }
EMSCRIPTEN_KEEPALIVE float* lidar_vxyz() { return vxyz.data(); }
EMSCRIPTEN_KEEPALIVE int32_t* lidar_vlabel() { return vlabel.data(); }

// 0 점 · 1 복셀 · 2 최적 가설 · 3 인라이어 수 · 4 지면 복셀 · 5 클러스터(셀 3개 이상) · 6 작은 조각
EMSCRIPTEN_KEEPALIVE int lidar_stat(int k) {
  const int v[] = {(int)pts.size(), (int)R.voxels.size(), R.best_hyp, R.best_score, R.n_ground, R.n_clusters, n_small};
  return k >= 0 && k < 7 ? v[k] : -1;
}
// 0 복셀 키 · 1 클러스터 라벨 배열 · 2 복셀 중심(float 비트)
EMSCRIPTEN_KEEPALIVE uint32_t lidar_hash(int k) { return k == 0 ? h_keys : k == 1 ? h_labels : h_centroids; }
// 0 복셀 · 1 RANSAC · 2 클러스터 · 3 합계 (ms)
EMSCRIPTEN_KEEPALIVE double lidar_ms(int k) { return k == 0 ? T.voxel : k == 1 ? T.ransac : k == 2 ? T.cluster : T.total; }
EMSCRIPTEN_KEEPALIVE float lidar_plane(int k) { return k == 0 ? PL.a : k == 1 ? PL.b : k == 2 ? PL.c : PL.d; }

}  // extern "C"
