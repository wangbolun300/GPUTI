#pragma once
#include<utility>
#include<array>
#include <limits>
#include <gputi/CType.hpp>
//#define GPUTI_USE_DOUBLE_PRECISION
// #define GPUTI_SHOW_INFO 
// TODO next when spliting time intervals, check if overlaps the current toi, then decide if we push it into the heap
// the reason of considerting it is that the limited heap size.

static const int HEAP_SIZE=1000;


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
__device__ Numccd& operator=(const Numccd& x)
    {
        if (this == &x)
            return *this;
        
        first=x.first;
        second=x.second;
        
        return *this;
    }

};

class Singleinterval{
public:
__device__ Singleinterval(){};
__device__ Singleinterval(Numccd f, Numccd s);
Numccd first;
Numccd second;
__device__ Singleinterval& operator=(const Singleinterval& x)
    {
        if (this == &x)
            return *this;
        first=x.first;
        second=x.second;
        return *this;
    }
};

// typedef std::pair<long, int> Numccd; //<k,n> pair present a number k/pow(2,n)
// typedef std::pair<Numccd, Numccd>
//         Singleinterval; // a interval presented by two double numbers
//typedef Singleinterval Interval3[3]; // 3 dimesional interval
class VectorMax3d{
public:
    __device__ __host__ VectorMax3d(){};
    __device__ __host__ VectorMax3d(Scalar a, Scalar b, Scalar c);
   
    Scalar v[3];
__device__  __host__  friend VectorMax3d operator+(const VectorMax3d &x, const VectorMax3d &y)
        {
            VectorMax3d out;
            out.v[0]=x.v[0]+y.v[0];
            out.v[1]=x.v[1]+y.v[1];
            out.v[2]=x.v[2]+y.v[2];
            return out;
        }
__device__  __host__  friend VectorMax3d operator-(const VectorMax3d &x, const VectorMax3d &y)
        {
            VectorMax3d out;
            out.v[0]=x.v[0]-y.v[0];
            out.v[1]=x.v[1]-y.v[1];
            out.v[2]=x.v[2]-y.v[2];
            return out;
        }
__device__ __host__  VectorMax3d& operator=(const VectorMax3d& x)
    {
        if (this == &x)
            return *this;
        v[0]=x.v[0];
        v[1]=x.v[1];
        v[2]=x.v[2];
        return *this;
    }
};

class interval_pair{
    public:
    __device__ interval_pair(const Singleinterval& a, const Singleinterval& b);
    __device__ interval_pair(){};
    Singleinterval first;
    Singleinterval second;
};

//typedef Scalar Vector3d[3];


typedef int ptest[5];

class CCDdata{
public:
    __host__ __device__ CCDdata(){};
    //CCDdata(const std::array<std::array<Scalar,3>,8>&input);
    Scalar v0s[3];
    Scalar v1s[3];
    Scalar v2s[3];
    Scalar v3s[3];
    Scalar v0e[3];
    Scalar v1e[3];
    Scalar v2e[3];
    Scalar v3e[3];
    bool is_edge;
};

CCDdata array_to_ccd(std::array<std::array<Scalar,3>,8> a, bool is_edge);
__device__ void single_test_wrapper(CCDdata* vfdata, bool &result);
void print_vector(Scalar* v, int size);
void print_vector(int* v, int size);

// for interval b = a;
__device__ void interval_cp(const Singleinterval& a,Singleinterval& b);
// CCDdata::CCDdata(const std::array<std::array<Scalar,3>,8>& input){
//     for(int i=0;i<3;i++){
//         v0s[i]=input[0][i];
//         v1s[i]=input[1][i];
//         v2s[i]=input[2][i];
//         v3s[i]=input[3][i];
//         v0e[i]=input[4][i];
//         v1e[i]=input[5][i];
//         v2e[i]=input[6][i];
//         v3e[i]=input[7][i];
//     }
// }
