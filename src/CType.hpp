#pragma once
#ifdef GPUTI_USE_DOUBLE_PRECISION
typedef double Scalar; 
#define SCALAR_LIMIT DBL_MAX;
#else
typedef float Scalar; 
#define SCALAR_LIMIT INT_MAX;
#endif