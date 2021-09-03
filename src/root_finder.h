#pragma once
#include<gputi/queue.h>

__device__ bool vertexFaceCCD_double(
    const CCDdata &data_in,
    const Scalar* err,
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag);

__device__ bool edgeEdgeCCD_double(
    const CCDdata &data_in,
    const Scalar* err,
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag);

__device__ __host__ void get_numerical_error(
    const VectorMax3d *vertices, const int vsize,
    const bool &check_vf,
    const bool using_minimum_separation,
    Scalar *error);
//__device__ 


