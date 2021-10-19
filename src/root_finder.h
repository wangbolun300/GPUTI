#pragma once
#include <gputi/Type.hpp>


__device__ __host__ void get_numerical_error(
    const VectorMax3d *vertices, const int vsize,
    const bool &check_vf,
    const bool using_minimum_separation,
    Scalar *error);

__device__ bool vertexFaceCCD(CCDdata *data_in,var_wrapper *vars);


