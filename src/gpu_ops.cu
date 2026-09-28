// GPU 구현. 버퍼는 생성자에서 한 번만 잡는다(프레임마다 cudaMalloc 하지 않는 게 실제 탑재 조건).
#include "gpu_ops.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <cstdlib>

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "CUDA %s @ %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

static constexpr int TPB = 256;
static inline int blocks(int n) { return (n + TPB - 1) / TPB; }

// ── 복셀: thrust 경로 ─────────────────────────────────────────────────────
__global__ void k_keys(const P4* pts, int n, Params p, uint32_t* keys, float4* vals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  P4 q = pts[i];
  keys[i] = voxel_key(q.x, q.y, q.z, p);
  vals[i] = make_float4(q.x, q.y, q.z, 1.f);
}

struct F4Add {
  __host__ __device__ float4 operator()(const float4& a, const float4& b) const {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
};

// ── 복셀: 직접 쓴 해시 경로 (open addressing + atomicCAS) ─────────────────
__device__ __forceinline__ uint32_t hash32(uint32_t k) {
  k ^= k >> 16; k *= 0x7feb352du; k ^= k >> 15; k *= 0x846ca68bu; k ^= k >> 16;
  return k;
}

__global__ void k_hash_insert(const P4* pts, int n, Params p, uint32_t* tk, float4* ts,
                              uint32_t mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  P4 q = pts[i];
  uint32_t key = voxel_key(q.x, q.y, q.z, p);
  if (key == EMPTY_KEY) return;
  uint32_t h = hash32(key) & mask;
  while (true) {
    uint32_t prev = atomicCAS(&tk[h], EMPTY_KEY, key);
    if (prev == EMPTY_KEY || prev == key) {
      atomicAdd(&ts[h].x, q.x);
      atomicAdd(&ts[h].y, q.y);
      atomicAdd(&ts[h].z, q.z);
      atomicAdd(&ts[h].w, 1.f);
      return;
    }
    h = (h + 1) & mask;  // 선형 탐사
  }
}

__global__ void k_hash_compact(const uint32_t* tk, const float4* ts, int tsize, uint32_t* okeys,
                               float4* ovals, int* nout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tsize) return;
  uint32_t k = tk[i];
  if (k == EMPTY_KEY) return;
  int o = atomicAdd(nout, 1);
  okeys[o] = k;
  ovals[o] = ts[i];
}

__global__ void k_pack(const uint32_t* okeys, const float4* ovals, int nv, Voxel* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nv) return;
  float4 s = ovals[i];
  out[i] = {okeys[i], s.x / s.w, s.y / s.w, s.z / s.w};
}

// ── RANSAC: 가설 하나 = 블록 하나, 블록 안 스레드가 점을 나눠 센 뒤 공유메모리 리덕션 ──
__global__ void k_ransac(const Voxel* v, int n, const int* hyp, Params p, int* score) {
  __shared__ Plane pl;
  __shared__ int ok;
  __shared__ int red[TPB];
  const int h = blockIdx.x, t = threadIdx.x;
  if (t == 0) {
    Voxel a = v[hyp[3 * h]], b = v[hyp[3 * h + 1]], c = v[hyp[3 * h + 2]];
    ok = plane3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, p.min_nz, pl);
  }
  __syncthreads();
  if (!ok) {  // ok는 공유값이라 블록 전체가 같이 빠진다
    if (t == 0) score[h] = -1;
    return;
  }
  int cnt = 0;
  for (int i = t; i < n; i += blockDim.x) cnt += is_inlier(pl, v[i].x, v[i].y, v[i].z, p.ransac_thresh);
  red[t] = cnt;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  if (t == 0) score[h] = red[0];
}

__global__ void k_argmax(const int* score, int iters, int* best) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int bh = -1, bs = -1;
  for (int h = 0; h < iters; ++h)
    if (score[h] > bs) { bs = score[h]; bh = h; }  // CPU와 같은 동점 규칙
  best[0] = bh; best[1] = bs;
}

__global__ void k_ground_occ(const Voxel* v, int n, const int* hyp, const int* best, Params p,
                             uint8_t* occ, int* n_ground) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Plane pl{0, 0, 1, 0};
  bool ok = false;
  int bh = best[0];
  if (bh >= 0) {
    Voxel a = v[hyp[3 * bh]], b = v[hyp[3 * bh + 1]], c = v[hyp[3 * bh + 2]];
    ok = plane3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, p.min_nz, pl);
  }
  Voxel q = v[i];
  if (ok && is_inlier(pl, q.x, q.y, q.z, p.ransac_thresh)) {
    atomicAdd(n_ground, 1);
    return;
  }
  int cell = bev_cell(q.x, q.y, p);
  if (cell >= 0) occ[cell] = 1;
}

// ── 연결 성분 A: 유니온-파인드 (atomicMin으로 작은 인덱스 쪽에 붙인다) ─────────
__device__ __forceinline__ int uf_find(int* L, int x) {
  volatile int* V = L;
  while (true) {
    int y = V[x];
    if (y == x) return x;
    x = y;
  }
}

__device__ void uf_union(int* L, int a, int b) {
  bool done = false;
  do {
    a = uf_find(L, a);
    b = uf_find(L, b);
    if (a < b) {
      int old = atomicMin(&L[b], a);
      done = (old == b);
      b = old;
    } else if (b < a) {
      int old = atomicMin(&L[a], b);
      done = (old == a);
      a = old;
    } else {
      done = true;
    }
  } while (!done);
}

__global__ void k_label_init(const uint8_t* occ, int N, int* L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) L[i] = occ[i] ? i : -1;
}

__global__ void k_uf_merge(const uint8_t* occ, int W, int H, int* L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= W * H || !occ[i]) return;
  int x = i % W, y = i / W;
  // 8-연결을 한 번씩만 보도록 "앞쪽" 이웃 4개만 합친다
  const int dx[4] = {-1, -1, 0, 1}, dy[4] = {0, -1, -1, -1};
  for (int k = 0; k < 4; ++k) {
    int nx = x + dx[k], ny = y + dy[k];
    if (nx < 0 || ny < 0 || nx >= W) continue;
    int j = ny * W + nx;
    if (occ[j]) uf_union(L, i, j);
  }
}

__global__ void k_uf_compress(const uint8_t* occ, int N, int* L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && occ[i]) L[i] = uf_find(L, i);
}

// ── 연결 성분 B: 반복 라벨 전파 (이웃 최소값, 수렴할 때까지) ────────────────
__global__ void k_prop(const uint8_t* occ, int W, int H, const int* in, int* out, int* changed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= W * H) return;
  if (!occ[i]) { out[i] = -1; return; }
  int x = i % W, y = i / W, m = in[i];
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      int nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      int j = ny * W + nx;
      if (occ[j] && in[j] < m) m = in[j];
    }
  out[i] = m;
  if (m != in[i]) *changed = 1;
}

__global__ void k_sizes(const uint8_t* occ, int N, const int* L, int* sz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && occ[i]) atomicAdd(&sz[L[i]], 1);
}

__global__ void k_count(const int* L, const int* sz, int N, int min_cells, int* ncl) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && L[i] == i && sz[i] >= min_cells) atomicAdd(ncl, 1);
}

// ────────────────────────────────────────────────────────────────────────
struct GpuPipeline::Impl {
  Params p;
  int cap = 0, tsize = 0, N = 0;
  cudaStream_t s;
  cudaEvent_t ev[6];
  cudaDeviceProp prop;
  P4 *h_pts = nullptr, *d_pts = nullptr;
  uint32_t *d_keys = nullptr, *d_okeys = nullptr, *d_tk = nullptr;
  float4 *d_vals = nullptr, *d_ovals = nullptr, *d_ts = nullptr;
  Voxel* d_vox = nullptr;
  int *d_hyp = nullptr, *d_score = nullptr, *d_best = nullptr, *d_ng = nullptr;
  int *d_L = nullptr, *d_L2 = nullptr, *d_sz = nullptr, *d_ncl = nullptr, *d_chg = nullptr,
      *d_nout = nullptr;
  uint8_t* d_occ = nullptr;
  int* h_small = nullptr;  // pinned 소형 결과 [best_h, best_s, n_ground, n_clusters, nv, changed]
};

GpuPipeline::GpuPipeline(const Params& p, int max_points) : d(new Impl) {
  d->p = p;
  d->cap = max_points;
  d->tsize = 1;
  while (d->tsize < 2 * max_points) d->tsize <<= 1;  // 부하율 0.5 이하
  d->N = p.gw * p.gh;
  CK(cudaGetDeviceProperties(&d->prop, 0));
  CK(cudaStreamCreate(&d->s));
  for (auto& e : d->ev) CK(cudaEventCreate(&e));
  CK(cudaMallocHost(&d->h_pts, sizeof(P4) * max_points));
  CK(cudaMallocHost(&d->h_small, sizeof(int) * 8));
  CK(cudaMalloc(&d->d_pts, sizeof(P4) * max_points));
  CK(cudaMalloc(&d->d_keys, sizeof(uint32_t) * max_points));
  CK(cudaMalloc(&d->d_vals, sizeof(float4) * max_points));
  CK(cudaMalloc(&d->d_okeys, sizeof(uint32_t) * max_points));
  CK(cudaMalloc(&d->d_ovals, sizeof(float4) * max_points));
  CK(cudaMalloc(&d->d_tk, sizeof(uint32_t) * d->tsize));
  CK(cudaMalloc(&d->d_ts, sizeof(float4) * d->tsize));
  CK(cudaMalloc(&d->d_vox, sizeof(Voxel) * max_points));
  CK(cudaMalloc(&d->d_hyp, sizeof(int) * 3 * p.ransac_iters));
  CK(cudaMalloc(&d->d_score, sizeof(int) * p.ransac_iters));
  CK(cudaMalloc(&d->d_best, sizeof(int) * 2));
  CK(cudaMalloc(&d->d_ng, sizeof(int)));
  CK(cudaMalloc(&d->d_nout, sizeof(int)));
  CK(cudaMalloc(&d->d_occ, d->N));
  CK(cudaMalloc(&d->d_L, sizeof(int) * d->N));
  CK(cudaMalloc(&d->d_L2, sizeof(int) * d->N));
  CK(cudaMalloc(&d->d_sz, sizeof(int) * d->N));
  CK(cudaMalloc(&d->d_ncl, sizeof(int)));
  CK(cudaMalloc(&d->d_chg, sizeof(int)));
}

GpuPipeline::~GpuPipeline() {
  cudaFreeHost(d->h_pts); cudaFreeHost(d->h_small);
  cudaFree(d->d_pts); cudaFree(d->d_keys); cudaFree(d->d_vals); cudaFree(d->d_okeys);
  cudaFree(d->d_ovals); cudaFree(d->d_tk); cudaFree(d->d_ts); cudaFree(d->d_vox);
  cudaFree(d->d_hyp); cudaFree(d->d_score); cudaFree(d->d_best); cudaFree(d->d_ng);
  cudaFree(d->d_nout); cudaFree(d->d_occ); cudaFree(d->d_L); cudaFree(d->d_L2);
  cudaFree(d->d_sz); cudaFree(d->d_ncl); cudaFree(d->d_chg);
  for (auto& e : d->ev) cudaEventDestroy(e);
  cudaStreamDestroy(d->s);
  delete d;
}

const char* GpuPipeline::device_name() const { return d->prop.name; }

StageMs GpuPipeline::run(const std::vector<P4>& pts, uint32_t seed, VoxelMode vm, ClusterMode cm,
                         bool validate, Result& r) {
  Impl& g = *d;
  const Params& p = g.p;
  const int n = (int)pts.size();
  cudaStream_t s = g.s;
  auto pol = thrust::cuda::par.on(s);
  StageMs t;

  // 센서 드라이버가 호스트 메모리에 놓은 스캔을 pinned 버퍼로 옮기는 비용은 측정 밖(드라이버 몫)
  std::copy(pts.begin(), pts.end(), g.h_pts);
  double w0 = now_ms();

  CK(cudaEventRecord(g.ev[0], s));
  CK(cudaMemcpyAsync(g.d_pts, g.h_pts, sizeof(P4) * n, cudaMemcpyHostToDevice, s));
  CK(cudaEventRecord(g.ev[1], s));

  // 1) 복셀
  int nv = 0;
  if (vm == VoxelMode::ThrustSort) {
    k_keys<<<blocks(n), TPB, 0, s>>>(g.d_pts, n, p, g.d_keys, g.d_vals);
    thrust::device_ptr<uint32_t> k(g.d_keys), ok(g.d_okeys);
    thrust::device_ptr<float4> v(g.d_vals), ov(g.d_ovals);
    thrust::sort_by_key(pol, k, k + n, v);
    auto e = thrust::reduce_by_key(pol, k, k + n, v, ok, ov, thrust::equal_to<uint32_t>(), F4Add());
    nv = (int)(e.first - ok);
    uint32_t last = 0;
    if (nv > 0) {
      CK(cudaMemcpyAsync(&last, g.d_okeys + nv - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
      CK(cudaStreamSynchronize(s));
      if (last == EMPTY_KEY) --nv;  // ROI 밖 묶음은 정렬상 맨 끝
    }
  } else {
    CK(cudaMemsetAsync(g.d_tk, 0xFF, sizeof(uint32_t) * g.tsize, s));
    CK(cudaMemsetAsync(g.d_ts, 0, sizeof(float4) * g.tsize, s));
    CK(cudaMemsetAsync(g.d_nout, 0, sizeof(int), s));
    k_hash_insert<<<blocks(n), TPB, 0, s>>>(g.d_pts, n, p, g.d_tk, g.d_ts, (uint32_t)(g.tsize - 1));
    k_hash_compact<<<blocks(g.tsize), TPB, 0, s>>>(g.d_tk, g.d_ts, g.tsize, g.d_okeys, g.d_ovals,
                                                   g.d_nout);
    CK(cudaMemcpyAsync(&g.h_small[4], g.d_nout, sizeof(int), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    nv = g.h_small[4];
    // 해시 순서는 실행마다 달라서, RANSAC 가설이 같은 복셀을 가리키도록 key로 정렬한다
    thrust::device_ptr<uint32_t> ok(g.d_okeys);
    thrust::device_ptr<float4> ov(g.d_ovals);
    thrust::sort_by_key(pol, ok, ok + nv, ov);
  }
  if (nv > 0) k_pack<<<blocks(nv), TPB, 0, s>>>(g.d_okeys, g.d_ovals, nv, g.d_vox);
  CK(cudaEventRecord(g.ev[2], s));

  // 2) RANSAC
  auto hyp = make_hypotheses(nv, p.ransac_iters, seed);
  CK(cudaMemcpyAsync(g.d_hyp, hyp.data(), sizeof(int) * hyp.size(), cudaMemcpyHostToDevice, s));
  k_ransac<<<p.ransac_iters, TPB, 0, s>>>(g.d_vox, nv, g.d_hyp, p, g.d_score);
  k_argmax<<<1, 32, 0, s>>>(g.d_score, p.ransac_iters, g.d_best);
  CK(cudaEventRecord(g.ev[3], s));

  // 3) 지면 제외 → 점유 격자 → 연결 성분
  CK(cudaMemsetAsync(g.d_occ, 0, g.N, s));
  CK(cudaMemsetAsync(g.d_ng, 0, sizeof(int), s));
  if (nv > 0)
    k_ground_occ<<<blocks(nv), TPB, 0, s>>>(g.d_vox, nv, g.d_hyp, g.d_best, p, g.d_occ, g.d_ng);
  k_label_init<<<blocks(g.N), TPB, 0, s>>>(g.d_occ, g.N, g.d_L);
  int iters = 0;
  if (cm == ClusterMode::UnionFind) {
    k_uf_merge<<<blocks(g.N), TPB, 0, s>>>(g.d_occ, p.gw, p.gh, g.d_L);
    k_uf_compress<<<blocks(g.N), TPB, 0, s>>>(g.d_occ, g.N, g.d_L);
  } else {
    // 수렴 플래그를 호스트가 읽을 때마다 동기화 왕복이 생긴다. Propagate16은 16회에 한 번만 읽는다
    // 플래그는 반복마다 0으로 지우므로 묶음의 마지막 반복만 반영한다. 거기서 변화가 없으면 수렴이고,
    // 수렴 뒤 더 돈 반복은 라벨을 바꾸지 않는다(최소값 전파는 멱등).
    const int every = (cm == ClusterMode::Propagate16) ? 16 : 1;
    while (true) {
      for (int k = 0; k < every; ++k) {
        CK(cudaMemsetAsync(g.d_chg, 0, sizeof(int), s));
        k_prop<<<blocks(g.N), TPB, 0, s>>>(g.d_occ, p.gw, p.gh, g.d_L, g.d_L2, g.d_chg);
        std::swap(g.d_L, g.d_L2);
        ++iters;
      }
      CK(cudaMemcpyAsync(&g.h_small[5], g.d_chg, sizeof(int), cudaMemcpyDeviceToHost, s));
      CK(cudaStreamSynchronize(s));
      if (!g.h_small[5]) break;
    }
  }
  CK(cudaMemsetAsync(g.d_sz, 0, sizeof(int) * g.N, s));
  CK(cudaMemsetAsync(g.d_ncl, 0, sizeof(int), s));
  k_sizes<<<blocks(g.N), TPB, 0, s>>>(g.d_occ, g.N, g.d_L, g.d_sz);
  k_count<<<blocks(g.N), TPB, 0, s>>>(g.d_L, g.d_sz, g.N, p.min_cells, g.d_ncl);
  CK(cudaEventRecord(g.ev[4], s));

  // 4) 요약값만 내린다(실제 탑재에서 다음 단계로 넘기는 양)
  CK(cudaMemcpyAsync(&g.h_small[0], g.d_best, sizeof(int) * 2, cudaMemcpyDeviceToHost, s));
  CK(cudaMemcpyAsync(&g.h_small[2], g.d_ng, sizeof(int), cudaMemcpyDeviceToHost, s));
  CK(cudaMemcpyAsync(&g.h_small[3], g.d_ncl, sizeof(int), cudaMemcpyDeviceToHost, s));
  CK(cudaEventRecord(g.ev[5], s));
  CK(cudaStreamSynchronize(s));
  CK(cudaGetLastError());
  double w1 = now_ms();

  float ms;
  CK(cudaEventElapsedTime(&ms, g.ev[0], g.ev[1])); t.h2d = ms;
  CK(cudaEventElapsedTime(&ms, g.ev[1], g.ev[2])); t.voxel = ms;
  CK(cudaEventElapsedTime(&ms, g.ev[2], g.ev[3])); t.ransac = ms;
  CK(cudaEventElapsedTime(&ms, g.ev[3], g.ev[4])); t.cluster = ms;
  CK(cudaEventElapsedTime(&ms, g.ev[4], g.ev[5])); t.d2h = ms;
  t.total = w1 - w0;  // 호스트 벽시계(동기화·런치 오버헤드 포함)

  r.best_hyp = g.h_small[0];
  r.best_score = g.h_small[1];
  r.n_ground = g.h_small[2];
  r.n_clusters = g.h_small[3];
  r.prop_iters = iters;
  if (validate) {
    r.voxels.resize(nv);
    CK(cudaMemcpy(r.voxels.data(), g.d_vox, sizeof(Voxel) * nv, cudaMemcpyDeviceToHost));
    r.labels.resize(g.N);
    CK(cudaMemcpy(r.labels.data(), g.d_L, sizeof(int) * g.N, cudaMemcpyDeviceToHost));
  }
  return t;
}
