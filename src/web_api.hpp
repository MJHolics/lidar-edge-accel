#pragma once
#include "common.hpp"

std::vector<P4> unpack_points(const uint16_t* q, int n, const Params& p);

extern "C" {
int lidar_run(const uint16_t* q, int n, uint32_t seed);
void lidar_set_iters(int n);
float* lidar_xyz();
int32_t* lidar_plabel();
float* lidar_vxyz();
int32_t* lidar_vlabel();
int lidar_stat(int k);
uint32_t lidar_hash(int k);
double lidar_ms(int k);
float lidar_plane(int k);
}
