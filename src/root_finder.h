#pragma once
#include <gputi/queue.h>
#include <vector>
#include <array>
#include <gputi/timer.hpp>
// #include <gputi/book.h>
namespace ccd
{
	__device__ void vertexFaceCCD(const CCDdata &data_in, const CCDConfig &config, CCDOut &out);
	__device__ void edgeEdgeCCD(const CCDdata &data_in, const CCDConfig &config, CCDOut &out);
	__device__ void vertexFaceMinimumSeparationCCD(const CCDdata &data_in, const CCDConfig &config, CCDOut &out);
	__device__ void edgeEdgeMinimumSeparationCCD(const CCDdata &data_in, const CCDConfig &config, CCDOut &out);
	void run_memory_pool_ccd(const std::vector<std::array<std::array<Scalar, 3>, 8>> &V, bool is_edge,
							 std::vector<int> &result_list, int parallel_nbr, double &runtime, Scalar &toi);

	// can be removed once device-only run_memory_pool_ccd copied over
	__global__ void initialize_memory_pool(MP_unit *units, int query_size);
	__global__ void compute_vf_tolerance_memory_pool(CCDdata *data, CCDConfig *config, const int query_size);
	__global__ void shift_queue_pointers(CCDConfig *config);
	__global__ void vf_ccd_memory_pool(MP_unit *units, int query_size, CCDdata *data, CCDConfig *config, int *results);
} // namespace ccd