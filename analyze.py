"""bench 결과 CSV 집계: 구현별 단계 지연(스캔별 중앙값의 분포)과 일치 검증 요약.

사용: python analyze.py results/run1
"""
import sys

import pandas as pd

prefix = sys.argv[1] if len(sys.argv) > 1 else "results/run1"
t = pd.read_csv(f"{prefix}_timing.csv")
v = pd.read_csv(f"{prefix}_validate.csv")

# 스캔마다 반복의 중앙값을 먼저 잡고(잡음 제거), 스캔 분포의 중앙값·p95·최대를 본다
per_scan = t.groupby(["impl", "scan"]).median(numeric_only=True).reset_index()
stages = ["h2d", "voxel", "ransac", "cluster", "d2h", "total"]
order = ["cpu_t1", "cpu_t2", "cpu_t4", "cpu_t8", "cpu_t16", "gpu_thrust_uf", "gpu_hash_uf", "gpu_hash_prop", "gpu_hash_prop16"]

rows = []
for impl in order:
    s = per_scan[per_scan.impl == impl]
    if s.empty:
        continue
    row = {"impl": impl}
    for c in stages:
        row[c] = s[c].median()
    row["total_p95"] = s["total"].quantile(0.95)
    row["total_max"] = s["total"].max()
    row["prop_iters_med"] = s["prop_iters"].median()
    rows.append(row)
summary = pd.DataFrame(rows).set_index("impl")
base = summary.loc["cpu_t1", "total"]
summary["speedup_vs_t1"] = base / summary["total"]
pd.set_option("display.width", 200)
print("== 지연(ms, 스캔별 중앙값의 중앙값) ==")
print(summary.round(3).to_string())

print("\n== 입력 규모 ==")
ref = v[v.impl == "cpu_t1"]
print(f"스캔 {t.scan.nunique()}장 · 점 {t.n_points.median():.0f}(중앙) · 복셀 {ref.n_voxels.median():.0f} · "
      f"지면 {ref.n_ground.median():.0f} · 클러스터 {ref.n_clusters.median():.0f}")

print("\n== CPU 직렬 대비 일치 검증 (스캔 수 기준) ==")
for impl in order[1:]:
    s = v[v.impl == impl]
    if s.empty:
        continue
    print(f"{impl:14s} 복셀수≠{(s.voxel_count_diff != 0).sum():3d} 키≠{(s.key_mismatch > 0).sum():3d} "
          f"중심최대오차={s.max_centroid_diff.max():.2e} 최적가설≠{(s.best_hyp_same == 0).sum():3d} "
          f"인라이어수≠{(s.score_diff != 0).sum():3d}(최대 {s.score_diff.abs().max()}) "
          f"지면수≠{(s.ground_diff != 0).sum():3d} 라벨≠{(s.label_mismatch > 0).sum():3d} "
          f"클러스터수≠{(s.cluster_diff != 0).sum():3d}")

summary.to_csv(f"{prefix}_summary.csv")
