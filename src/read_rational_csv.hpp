#pragma once

#include <vector>
#include <array>
// #include <gputi/CType.hpp>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>

namespace ccd
{

#ifdef GPUTI_USE_DOUBLE_PRECISION
	typedef double3 Scalar3;
	typedef double2 Scalar2;
	typedef double Scalar;
	__host__ __device__ Scalar3 make_Scalar3(const Scalar a, const Scalar b,
											 const Scalar &c);
	__host__ __device__ Scalar2 make_Scalar2(const Scalar a, const Scalar b);
#warning Using Double
#define SCALAR_LIMIT DBL_MAX;
#else
	typedef float3 Scalar3;
	typedef float2 Scalar2;
	typedef float Scalar;
#warning Using Float
	__host__ __device__ Scalar3 make_Scalar3(const Scalar a, const Scalar b,
											 const Scalar &c);
	__host__ __device__ Scalar2 make_Scalar2(const Scalar a, const Scalar b);
#define SCALAR_LIMIT INT_MAX;
#endif

	std::vector<std::array<ccd::Scalar, 3>>
	read_rational_csv(const std::string &inputFileName, std::vector<bool> &results);

} // namespace ccd