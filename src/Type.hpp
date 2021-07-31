#pragma once
#include<utility>
#include<array>

typedef float Scalar; 
class Numccd{
public:
__device__ Numccd(){};
long first;
int second;
};

class Singleinterval{
public:
__device__ Singleinterval(){};
__device__ Singleinterval(Numccd f, Numccd s);
Numccd first;
Numccd second;
};

// typedef std::pair<long, int> Numccd; //<k,n> pair present a number k/pow(2,n)
// typedef std::pair<Numccd, Numccd>
//         Singleinterval; // a interval presented by two double numbers
//typedef Singleinterval Interval3[3]; // 3 dimesional interval
class VectorMax3d{
public:
    __device__ VectorMax3d(){};
    Scalar v[3];
};
//typedef Scalar Vector3d[3];

static const int HEAP_SIZE=1000;
typedef int ptest[5];