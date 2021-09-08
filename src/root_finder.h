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

__device__ bool CCD_Solver(
    const CCDdata &data_in,
    const Scalar err[3],
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag,
    bool is_vf);
//__device__ 
// __device__ void single_test_wrapper_return_toi(CCDdata *data, bool &result, Scalar &time_impact);

