#pragma once
#include<utility>
#include<array>

typedef float Scalar; 
typedef std::pair<long, int> Numccd; //<k,n> pair present a number k/pow(2,n)
typedef std::pair<Numccd, Numccd>
        Singleinterval; // a interval presented by two double numbers
typedef Singleinterval Interval3[3]; // 3 dimesional interval
class VectorMax3d{
public:
    __device__ VectorMax3d();
    Scalar v[3];
};

static const int HEAP_SIZE=1000;
typedef int ptest[5];