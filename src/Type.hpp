#pragma once
#include<utility>
#include<array>
#include <limits>
#include <gputi/CType.hpp>
//#define GPUTI_USE_DOUBLE_PRECISION
// #define GPUTI_SHOW_INFO 
// TODO next when spliting time intervals, check if overlaps the current toi, then decide if we push it into the heap
// the reason of considerting it is that the limited heap size.
// token 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
static const int HEAP_SIZE=1000;

// overflow instructions
static const int NO_OVERFLOW = 0;
static const int BISECTION_OVERFLOW = 1;
static const int HEAP_OVERFLOW = 2;
static const int ITERATION_OVERFLOW = 3;


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
   __device__ __host__ void init(Scalar a, Scalar b, Scalar c);
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

class HPvar{
public:


};
class ERRvar{
public:
VectorMax3d v0s;
VectorMax3d v1s;
VectorMax3d v2s;
VectorMax3d v3s;
VectorMax3d v0e;
VectorMax3d v1e;
VectorMax3d v2e;
VectorMax3d v3e;
VectorMax3d p000;
VectorMax3d p001;
VectorMax3d p011;
VectorMax3d p010;
VectorMax3d p100;
VectorMax3d p101;
VectorMax3d p111;
VectorMax3d p110;
Scalar dl;
Scalar edge0_length;
Scalar edge1_length;
Scalar m_r;
Scalar m_r1; 
Scalar m_temp; 
int  m_i;

VectorMax3d res;

Scalar eefilter;
Scalar vffilter;
Scalar xmax;
Scalar ymax;
Scalar zmax;
Scalar delta_x;
Scalar delta_y;
Scalar delta_z;
int itr_err;

};

class SOLVEvar{
public:


};
class CCDvar{
public:
bool res;
ERRvar errvar;
SOLVEvar solvar;
};


