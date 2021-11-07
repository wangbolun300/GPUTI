#pragma once
#include<gputi/queue.h>

namespace ccd{
__device__ void vertexFaceCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);
__device__ void edgeEdgeCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);
__device__ void vertexFaceMinimumSeparationCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);
__device__ void edgeEdgeMinimumSeparationCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);

}