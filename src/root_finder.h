#pragma once
#include<gputi/queue.h>
#include<vector>
#include<array>
#include <gputi/timer.hpp>
// #include <gputi/book.h>
namespace ccd{
__device__ void vertexFaceCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);
__device__ void edgeEdgeCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);
__device__ void vertexFaceMinimumSeparationCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);
__device__ void edgeEdgeMinimumSeparationCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);
void run_memory_pool_ccd(const std::vector<std::array<std::array<Scalar, 3>, 8>> &V, bool is_edge,
                 std::vector<bool> &result_list, int parallel_nbr, double &runtime);
}