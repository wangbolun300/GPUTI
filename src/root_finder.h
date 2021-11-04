#pragma once
#include<gputi/queue.h>

namespace ccd{
__device__ void vertexFaceCCD(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);


}