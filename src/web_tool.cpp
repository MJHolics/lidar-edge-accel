// 웹 화면용 준비 도구(네이티브).
//   web_tool pack <velodyne_dir> <out_dir> <start> <step> <count>
//       KITTI 스캔에서 ROI 안 점만 골라 uint16 3개로 줄여 f00.bin… 으로 쓴다. 원본 대비 결과 차이도 찍는다.
//   web_tool seeds <out_dir> <count> <nseed>
//       같은 입력에 RANSAC 시드만 바꿔 지면·클러스터 수가 얼마나 흔들리는지 CSV로 낸다(가설 256·1024·4096개).
//   web_tool ref <out_dir> <start> <step> <count> [reps=20]
//       줄인 파일을 web_api로 돌려 대조 정답(JSON)을 표준 출력으로 낸다. 브라우저가 같은 값을 내야 한다.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include "cpu_ops.hpp"
#include "web_api.hpp"

static std::string scan_path(const std::string& dir, int idx) {
  char b[32];
  std::snprintf(b, sizeof b, "/%010d.bin", idx);
  return dir + b;
}
static std::string frame_path(const std::string& dir, int k) {
  char b[32];
  std::snprintf(b, sizeof b, "/f%02d.bin", k);
  return dir + b;
}

static std::vector<uint16_t> pack(const std::vector<P4>& pts, const Params& p) {
  std::vector<uint16_t> q;
  q.reserve(pts.size() * 3);
  auto quant = [](float v, float lo, float hi) {
    int k = (int)floorf((v - lo) / (hi - lo) * 65536.f);
    return (uint16_t)std::min(65535, std::max(0, k));
  };
  for (const P4& a : pts) {
    if (voxel_key(a.x, a.y, a.z, p) == EMPTY_KEY) continue;      // ROI 밖은 어차피 첫 단계에서 버려진다
    q.push_back(quant(a.x, p.xmin, p.xmax));
    q.push_back(quant(a.y, p.ymin, p.ymax));
    q.push_back(quant(a.z, p.zmin, p.zmax));
  }
  return q;
}

static std::vector<uint16_t> read_u16(const std::string& path) {
  std::vector<uint16_t> v;
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(sz / 2);
    if (std::fread(v.data(), 2, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
  }
  return v;
}

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: web_tool pack|ref ...\n"); return 1; }
  Params p;
  std::string mode = argv[1];
  if (mode == "pack") {
    std::string in = argv[2], out = argv[3];
    int start = std::atoi(argv[4]), step = std::atoi(argv[5]), count = std::atoi(argv[6]);
    for (int k = 0; k < count; ++k) {
      int idx = start + k * step;
      auto pts = load_bin(scan_path(in, idx));
      if (pts.empty()) { std::fprintf(stderr, "no scan %d\n", idx); return 1; }
      auto q = pack(pts, p);
      FILE* f = std::fopen(frame_path(out, k).c_str(), "wb");
      std::fwrite(q.data(), 2, q.size(), f);
      std::fclose(f);
      Result a, b;
      uint32_t seed = 1000u + (uint32_t)idx;
      run_cpu(pts, p, 1, seed, a);
      run_cpu(unpack_points(q.data(), (int)q.size() / 3, p), p, 1, seed, b);
      int lab = 0;
      for (size_t i = 0; i < a.labels.size(); ++i) lab += (a.labels[i] >= 0) != (b.labels[i] >= 0);
      std::fprintf(stderr, "scan %3d  점 %zu → %zu  복셀 %zu/%zu  지면 %d/%d  클러스터 %d/%d  점유 셀 차이 %d\n", idx, pts.size(),
                   q.size() / 3, a.voxels.size(), b.voxels.size(), a.n_ground, b.n_ground, a.n_clusters, b.n_clusters, lab);
    }
    return 0;
  }
  if (mode == "seeds") {                                         // 같은 입력에서 RANSAC 가설만 바꾸면 결과가 얼마나 흔들리나
    std::string dir = argv[2];
    int count = std::atoi(argv[3]), nseed = std::atoi(argv[4]);
    std::printf("frame,iters,seed,best_score,ground,clusters\n");
    for (int k = 0; k < count; ++k) {
      auto q = read_u16(frame_path(dir, k));
      auto pts = unpack_points(q.data(), (int)q.size() / 3, p);
      for (int iters : {256, 1024, 4096}) {
        Params pp = p;
        pp.ransac_iters = iters;
        for (int sd = 1; sd <= nseed; ++sd) {
          Result r;
          run_cpu(pts, pp, 1, (uint32_t)sd * 7919u, r);
          std::printf("%d,%d,%d,%d,%d,%d\n", k, iters, sd, r.best_score, r.n_ground, r.n_clusters);
        }
      }
    }
    return 0;
  }
  std::string dir = argv[2];
  int start = std::atoi(argv[3]), step = std::atoi(argv[4]), count = std::atoi(argv[5]);
  int reps = argc > 6 ? std::atoi(argv[6]) : 20;
  std::printf("{\"frames\":[");
  for (int k = 0; k < count; ++k) {
    int idx = start + k * step;
    auto q = read_u16(frame_path(dir, k));
    if (q.empty()) { std::fprintf(stderr, "no frame %d\n", k); return 1; }
    std::vector<double> tot, vox, ran, clu;
    for (int r = 0; r < reps; ++r) {
      lidar_run(q.data(), (int)q.size() / 3, 1000u + (uint32_t)idx);
      vox.push_back(lidar_ms(0)); ran.push_back(lidar_ms(1)); clu.push_back(lidar_ms(2)); tot.push_back(lidar_ms(3));
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    std::printf("%s{\"file\":\"f%02d.bin\",\"scan\":%d,\"seed\":%u,\"points\":%d,\"voxels\":%d,\"best_hyp\":%d,\"best_score\":%d,"
                "\"ground\":%d,\"clusters\":%d,\"small\":%d,\"h_keys\":%u,\"h_labels\":%u,\"h_centroids\":%u,"
                "\"native_ms\":{\"voxel\":%.4f,\"ransac\":%.4f,\"cluster\":%.4f,\"total\":%.4f}}",
                k ? "," : "", k, idx, 1000u + (uint32_t)idx, lidar_stat(0), lidar_stat(1), lidar_stat(2), lidar_stat(3), lidar_stat(4),
                lidar_stat(5), lidar_stat(6), lidar_hash(0), lidar_hash(1), lidar_hash(2), med(vox), med(ran), med(clu), med(tot));
  }
  std::printf("]}\n");
  return 0;
}
