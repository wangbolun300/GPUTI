#pragma once
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>

namespace ccd
{
#ifdef GPUTI_USE_DOUBLE_PRECISION
	typedef double3 Scalar3;
	typedef double2 Scalar2;
	typedef double Scalar;
#warning Using Double
#define SCALAR_LIMIT DBL_MAX;
#else
	typedef float3 Scalar3;
	typedef float2 Scalar2;
	typedef float Scalar;
#warning Using Float
#define make_Scalar3 make_float3
#define make_Scalar2 make_float2
#define SCALAR_LIMIT INT_MAX;
#endif

} // namespace ccd