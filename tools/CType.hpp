#pragma once
#include <cfloat>

namespace ccd{
#ifdef GPUTI_USE_DOUBLE_PRECISION
typedef double Scalar;
#warning using double
#define SCALAR_LIMIT DBL_MAX;
#else
typedef float Scalar; 
#warning using float
#define SCALAR_LIMIT INT_MAX;
#endif

}