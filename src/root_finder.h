#pragma once
#include<gputi/queue.h>
__device__ bool evaluate_bbox_one_dimension_vector_return_tolerance(
        Scalar* t_up,
        Scalar* t_dw,
        Scalar* u_up,
        Scalar* u_dw,
        Scalar* v_up,
        Scalar* v_dw,
        const Scalar* a0s,
        const Scalar* a1s,
        const Scalar* b0s,
        const Scalar* b1s,
        const Scalar* a0e,
        const Scalar* a1e,
        const Scalar* b0e,
        const Scalar* b1e,
        const int dimension,
        const bool check_vf,
        const Scalar eps,
        const Scalar ms,
        bool &bbox_in_eps,
        Scalar &tol);

__device__ void convert_tuv_to_array(
        const Singleinterval* itv0, Singleinterval* itv1, Singleinterval* itv2, 
        Scalar* t_up,
        Scalar* t_dw,
        Scalar* u_up,
        Scalar* u_dw,
        Scalar* v_up,
        Scalar* v_dw);
__device__ long power(const long a, const int b);
__device__ VectorMax3d width(const Singleinterval* x);
__device__ bool vertexFaceCCD_double(
    CCDvar &vars,
    CCDdata* data_in,
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
    CCDvar &vars,
    CCDdata* data_in,
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


