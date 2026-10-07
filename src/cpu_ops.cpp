// CPU 구현: 직렬(threads=1)과 OpenMP 멀티코어. 알고리즘은 GPU와 같고 병렬화 방식만 다르다.
#include "cpu_ops.hpp"
#include <algorithm>
#ifdef _OPENMP                       // -fopenmp 없이(WASM 등) 빌드하면 직렬 경로만 남는다
#include <parallel/algorithm>
#include <omp.h>
#endif

#ifndef FIX_SORT_THREADS
#define FIX_SORT_THREADS 1
#endif

// ── 복셀 다운샘플: (key, 점 인덱스) 정렬 후 같은 key 구간을 평균 ─────────────
std::vector<Voxel> voxel_cpu(const std::vector<P4>& pts, const Params& p, int threads) {
  // __gnu_parallel::sort는 num_threads 절이 아니라 omp_get_max_threads()를 따른다.
  // 이걸 맞추지 않으면 병렬 영역마다 팀 크기가 threads↔16으로 바뀌어 2~8스레드가 직렬보다 느려졌다.
#ifdef _OPENMP
  if (FIX_SORT_THREADS) omp_set_num_threads(threads);
#endif
  const int n = (int)pts.size();
  std::vector<std::pair<uint32_t, int>> kv(n);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int i = 0; i < n; ++i) kv[i] = {voxel_key(pts[i].x, pts[i].y, pts[i].z, p), i};

#ifdef _OPENMP
  if (threads > 1)
    __gnu_parallel::sort(kv.begin(), kv.end());
  else
#endif
    std::sort(kv.begin(), kv.end());

  // ROI 밖(EMPTY_KEY)은 정렬 후 맨 뒤에 모인다.
  int m = n;
  while (m > 0 && kv[m - 1].first == EMPTY_KEY) --m;

  std::vector<int> starts;
  starts.reserve(m / 2 + 1);
  for (int i = 0; i < m; ++i)
    if (i == 0 || kv[i].first != kv[i - 1].first) starts.push_back(i);
  starts.push_back(m);

  const int nv = (int)starts.size() - 1;
  std::vector<Voxel> out(nv);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int s = 0; s < nv; ++s) {
    float sx = 0, sy = 0, sz = 0;
    for (int j = starts[s]; j < starts[s + 1]; ++j) {
      const P4& q = pts[kv[j].second];
      sx += q.x; sy += q.y; sz += q.z;
    }
    float c = (float)(starts[s + 1] - starts[s]);
    out[s] = {kv[starts[s]].first, sx / c, sy / c, sz / c};
  }
  return out;
}

// ── RANSAC 지면: 가설마다 인라이어 수를 센다. 병렬은 가설 축으로 나눈다 ─────
void ransac_cpu(const std::vector<Voxel>& v, const std::vector<int>& hyp, const Params& p,
                int threads, Result& r) {
  const int iters = (int)hyp.size() / 3, n = (int)v.size();
  std::vector<int> score(iters, -1);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int h = 0; h < iters; ++h) {
    const Voxel &a = v[hyp[3 * h]], &b = v[hyp[3 * h + 1]], &c = v[hyp[3 * h + 2]];
    Plane pl;
    if (!plane3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, p.min_nz, pl)) continue;
    int cnt = 0;
    for (int i = 0; i < n; ++i) cnt += is_inlier(pl, v[i].x, v[i].y, v[i].z, p.ransac_thresh);
    score[h] = cnt;
  }
  r.best_hyp = -1; r.best_score = -1;
  for (int h = 0; h < iters; ++h)
    if (score[h] > r.best_score) { r.best_score = score[h]; r.best_hyp = h; }  // 동점이면 앞 가설
}

// ── 지면 제외 후 BEV 점유 격자 ──────────────────────────────────────────────
std::vector<uint8_t> occupancy_cpu(const std::vector<Voxel>& v, const std::vector<int>& hyp,
                                   const Params& p, Result& r) {
  std::vector<uint8_t> occ(p.gw * p.gh, 0);
  Plane pl{0, 0, 1, 0};
  bool ok = false;
  if (r.best_hyp >= 0) {
    const Voxel &a = v[hyp[3 * r.best_hyp]], &b = v[hyp[3 * r.best_hyp + 1]],
                &c = v[hyp[3 * r.best_hyp + 2]];
    ok = plane3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, p.min_nz, pl);
  }
  int ng = 0;
  for (const auto& q : v) {
    if (ok && is_inlier(pl, q.x, q.y, q.z, p.ransac_thresh)) { ++ng; continue; }
    int cell = bev_cell(q.x, q.y, p);
    if (cell >= 0) occ[cell] = 1;
  }
  r.n_ground = ng;
  return occ;
}

// ── 8-연결 성분: BFS. 인덱스 순으로 시작하므로 대표 = 성분 내 최소 인덱스 ────
void cluster_cpu(const std::vector<uint8_t>& occ, const Params& p, Result& r) {
  const int W = p.gw, H = p.gh, N = W * H;
  r.labels.assign(N, -1);
  std::vector<int> q;
  q.reserve(N);
  r.n_clusters = 0;
  for (int s = 0; s < N; ++s) {
    if (!occ[s] || r.labels[s] != -1) continue;
    q.clear();
    q.push_back(s);
    r.labels[s] = s;
    for (size_t k = 0; k < q.size(); ++k) {
      int c = q[k], cx = c % W, cy = c / W;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          int x = cx + dx, y = cy + dy;
          if (x < 0 || y < 0 || x >= W || y >= H) continue;
          int j = y * W + x;
          if (occ[j] && r.labels[j] == -1) { r.labels[j] = s; q.push_back(j); }
        }
    }
    if ((int)q.size() >= p.min_cells) ++r.n_clusters;
  }
}

StageMs run_cpu(const std::vector<P4>& pts, const Params& p, int threads, uint32_t seed, Result& r) {
  StageMs t;
  double t0 = now_ms();
  r.voxels = voxel_cpu(pts, p, threads);
  double t1 = now_ms();
  auto hyp = make_hypotheses((int)r.voxels.size(), p.ransac_iters, seed);
  ransac_cpu(r.voxels, hyp, p, threads, r);
  double t2 = now_ms();
  auto occ = occupancy_cpu(r.voxels, hyp, p, r);
  cluster_cpu(occ, p, r);
  double t3 = now_ms();
  t.voxel = t1 - t0; t.ransac = t2 - t1; t.cluster = t3 - t2; t.total = t3 - t0;
  return t;
}
