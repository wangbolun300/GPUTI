#pragma once
#include<utility>
#include<array>
#include <limits>
//#define GPUTI_USE_DOUBLE_PRECISION

#ifdef GPUTI_USE_DOUBLE_PRECISION
typedef double Scalar; 
#define SCALAR_LIMIT DBL_MAX;
#else
typedef float Scalar; 
#define SCALAR_LIMIT INT_MAX;
#endif


class Numccd{
public:
__device__ Numccd(){};
__device__ Numccd(const long a, const int b);
long first;
int second;
__device__ friend bool operator==(const Numccd &r, const Numccd &r1)
        {
            if(r.first==r1.first&&r.second==r1.second){
                return true;
            }
            return false;
        }
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

class interval_pair{
    public:
    __device__ interval_pair(const Singleinterval& a, const Singleinterval& b);
    __device__ interval_pair(){};
    Singleinterval first;
    Singleinterval second;
};

//typedef Scalar Vector3d[3];

static const int HEAP_SIZE=1000;
typedef int ptest[5];
