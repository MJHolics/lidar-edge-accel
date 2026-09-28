// 벤치마크 드라이버: KITTI 스캔마다 모든 구현을 돌리고, (1) CPU 직렬 결과와 일치하는지 (2) 단계별 시간을 CSV로 남긴다.
// 사용: bench <velodyne_dir> <out_prefix> [reps=5] [max_scans=0(전부)]
#include <algorithm>
#include <dirent.h>
#include <string>
#include <vector>
#include "common.hpp"
#include "cpu_ops.hpp"
#include "gpu_ops.hpp"

static std::vector<std::string> list_bins(const std::string& dir) {
  std::vector<std::string> v;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      std::string n = e->d_name;
      if (n.size() > 4 && n.substr(n.size() - 4) == ".bin") v.push_back(dir + "/" + n);
    }
    closedir(d);
  }
  std::sort(v.begin(), v.end());
  return v;
}

struct Cmp {
  int voxel_count_diff = 0, key_mismatch = 0;
  double max_centroid_diff = 0;
  int best_hyp_same = 1, score_diff = 0, ground_diff = 0, label_mismatch = 0, cluster_diff = 0;
};

static Cmp compare(const Result& ref, const Result& r) {
  Cmp c;
  c.voxel_count_diff = (int)r.voxels.size() - (int)ref.voxels.size();
  size_t m = std::min(r.voxels.size(), ref.voxels.size());
  for (size_t i = 0; i < m; ++i) {
    if (r.voxels[i].key != ref.voxels[i].key) { ++c.key_mismatch; continue; }
    double d = std::max({std::fabs(r.voxels[i].x - ref.voxels[i].x), std::fabs(r.voxels[i].y - ref.voxels[i].y),
                         std::fabs(r.voxels[i].z - ref.voxels[i].z)});
    c.max_centroid_diff = std::max(c.max_centroid_diff, d);
  }
  c.best_hyp_same = r.best_hyp == ref.best_hyp;
  c.score_diff = r.best_score - ref.best_score;
  c.ground_diff = r.n_ground - ref.n_ground;
  for (size_t i = 0; i < ref.labels.size() && i < r.labels.size(); ++i) c.label_mismatch += r.labels[i] != ref.labels[i];
  c.cluster_diff = r.n_clusters - ref.n_clusters;
  return c;
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: bench <velodyne_dir> <out_prefix> [reps] [max_scans]\n"); return 1; }
  std::string dir = argv[1], out = argv[2];
  int reps = argc > 3 ? std::atoi(argv[3]) : 5;
  int max_scans = argc > 4 ? std::atoi(argv[4]) : 0;
  auto files = list_bins(dir);
  if (max_scans > 0 && (int)files.size() > max_scans) files.resize(max_scans);
  if (files.empty()) { std::fprintf(stderr, "no .bin in %s\n", dir.c_str()); return 1; }

  Params p;
  int maxp = 0;
  std::vector<std::vector<P4>> scans;
  for (auto& f : files) { scans.push_back(load_bin(f)); maxp = std::max(maxp, (int)scans.back().size()); }
  GpuPipeline gpu(p, maxp);
  std::printf("scans=%zu max_points=%d gpu=%s\n", scans.size(), maxp, gpu.device_name());

  FILE* ft = std::fopen((out + "_timing.csv").c_str(), "w");
  FILE* fv = std::fopen((out + "_validate.csv").c_str(), "w");
  std::fprintf(ft, "scan,impl,rep,n_points,n_voxels,h2d,voxel,ransac,cluster,d2h,total,prop_iters\n");
  std::fprintf(fv, "scan,impl,voxel_count_diff,key_mismatch,max_centroid_diff,best_hyp_same,score_diff,ground_diff,label_mismatch,cluster_diff,n_voxels,n_ground,n_clusters\n");

  const int thread_set[] = {1, 2, 4, 8, 16};
  struct G { const char* name; VoxelMode vm; ClusterMode cm; };
  const G gimpl[] = {{"gpu_thrust_uf", VoxelMode::ThrustSort, ClusterMode::UnionFind},
                     {"gpu_hash_uf", VoxelMode::Hash, ClusterMode::UnionFind},
                     {"gpu_hash_prop", VoxelMode::Hash, ClusterMode::Propagate},
                     {"gpu_hash_prop16", VoxelMode::Hash, ClusterMode::Propagate16}};

  // 워밍업: 첫 CUDA 호출·thrust 임시버퍼 할당·OpenMP 스레드풀 생성을 측정에서 뺀다
  for (int w = 0; w < 3; ++w) {
    Result r;
    run_cpu(scans[0], p, 16, 1, r);
    for (auto& g : gimpl) gpu.run(scans[0], 1, g.vm, g.cm, false, r);
  }

  for (size_t si = 0; si < scans.size(); ++si) {
    const auto& pts = scans[si];
    uint32_t seed = 1000u + (uint32_t)si;
    Result ref;
    run_cpu(pts, p, 1, seed, ref);
    auto emit_v = [&](const char* name, const Result& r) {
      Cmp c = compare(ref, r);
      std::fprintf(fv, "%zu,%s,%d,%d,%.3g,%d,%d,%d,%d,%d,%zu,%d,%d\n", si, name, c.voxel_count_diff, c.key_mismatch,
                   c.max_centroid_diff, c.best_hyp_same, c.score_diff, c.ground_diff, c.label_mismatch,
                   c.cluster_diff, r.voxels.size(), r.n_ground, r.n_clusters);
    };
    for (int th : thread_set) {
      char name[32];
      std::snprintf(name, sizeof name, "cpu_t%d", th);
      Result r;
      run_cpu(pts, p, th, seed, r);
      emit_v(name, r);
      for (int k = 0; k < reps; ++k) {
        Result rr;
        StageMs t = run_cpu(pts, p, th, seed, rr);
        std::fprintf(ft, "%zu,%s,%d,%zu,%zu,0,%.4f,%.4f,%.4f,0,%.4f,0\n", si, name, k, pts.size(), rr.voxels.size(),
                     t.voxel, t.ransac, t.cluster, t.total);
      }
    }
    for (auto& g : gimpl) {
      Result r;
      gpu.run(pts, seed, g.vm, g.cm, true, r);
      emit_v(g.name, r);
      for (int k = 0; k < reps; ++k) {
        Result rr;
        StageMs t = gpu.run(pts, seed, g.vm, g.cm, false, rr);
        std::fprintf(ft, "%zu,%s,%d,%zu,%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", si, g.name, k, pts.size(),
                     r.voxels.size(), t.h2d, t.voxel, t.ransac, t.cluster, t.d2h, t.total, rr.prop_iters);
      }
    }
    std::fflush(ft);  // 스캔 단위로 즉시 저장(WSL VM이 도중에 죽어도 남도록)
    std::fflush(fv);
    if (si % 20 == 0) std::printf("scan %zu/%zu voxels=%zu clusters=%d\n", si, scans.size(), ref.voxels.size(), ref.n_clusters);
  }
  std::fclose(ft);
  std::fclose(fv);
  std::printf("done\n");
  return 0;
}
