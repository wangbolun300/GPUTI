#pragma once
#include<gputi/queue.h>


__device__ __host__ void get_numerical_error(
    const VectorMax3d *vertices, const int vsize,
    const bool &check_vf,
    const bool using_minimum_separation,
    Scalar *error);

__device__ bool vertexFaceCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out,MinHeap &istack);


