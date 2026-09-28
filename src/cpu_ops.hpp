#pragma once
#include "common.hpp"

std::vector<Voxel> voxel_cpu(const std::vector<P4>& pts, const Params& p, int threads);
void ransac_cpu(const std::vector<Voxel>& v, const std::vector<int>& hyp, const Params& p,
                int threads, Result& r);
std::vector<uint8_t> occupancy_cpu(const std::vector<Voxel>& v, const std::vector<int>& hyp,
                                   const Params& p, Result& r);
void cluster_cpu(const std::vector<uint8_t>& occ, const Params& p, Result& r);
StageMs run_cpu(const std::vector<P4>& pts, const Params& p, int threads, uint32_t seed, Result& r);
