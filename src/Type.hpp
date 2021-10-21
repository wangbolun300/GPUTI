#pragma once
#include<utility>
#include<array>
#include <limits>
#include <gputi/CType.hpp>
#include <assert.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
//#define GPUTI_USE_DOUBLE_PRECISION
// #define GPUTI_GO_DEAP_HEAP
static const int TESTING_ID = 219064;
static const int TEST_SIZE=1e6;
static const int TEST_NBR_QUERIES=1e9;// set as large as possible to avoid truncation of reading data
// #define CHECK_EE
#define NO_CHECK_MS
#define CALCULATE_ERROR_BOUND
#define TIME_UPPER_IS_ONE


// TODO next when spliting time intervals, check if overlaps the current toi, then decide if we push it into the heap
// the reason of considerting it is that the limited heap size.
// token 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


static const int HEAP_SIZE=1000;

// overflow instructions
static const int NO_OVERFLOW = 0;
static const int BISECTION_OVERFLOW = 1;
static const int HEAP_OVERFLOW = 2;
static const int ITERATION_OVERFLOW = 3;


class Singleinterval{
public:
__device__ Singleinterval(){};
__device__ Singleinterval(const Scalar &f, const Scalar &s);
Scalar first;
Scalar second;
__device__ Singleinterval& operator=(const Singleinterval& x)
    {
        if (this == &x)
            return *this;
        first=x.first;
        second=x.second;
        return *this;
    }
};

class interval_pair{
    public:
    __device__ interval_pair(const Singleinterval& itv);
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
    __device__ __host__  CCDdata& operator=(const CCDdata& x)
    {
        if (this == &x)
            return *this;
        for(int i=0;i<3;i++){
            v0s[i]=x.v0s[i];
            v1s[i]=x.v1s[i];
            v2s[i]=x.v2s[i];
            v3s[i]=x.v3s[i];
            v0e[i]=x.v0e[i];
            v1e[i]=x.v1e[i];
            v2e[i]=x.v2e[i];
            v3e[i]=x.v3e[i];
        }
        return *this;
    }
};

CCDdata array_to_ccd(std::array<std::array<Scalar,3>,8> a, bool is_edge);
__device__ void single_test_wrapper(CCDdata* vfdata, bool &result);
void print_vector(Scalar* v, int size);
void print_vector(int* v, int size);



class item {
public:
	int level;
	Singleinterval itv[3];
	__device__ item(const Singleinterval si[3], const int &level);
	__device__ item();
	__device__ item& operator=(const item& x)
    {
        if (this == &x)
            return *this;
        // itv[0]=x.itv[0];
		// itv[1]=x.itv[1];
		// itv[2]=x.itv[2];
    itv[0].first = x.itv[0].first;
    itv[0].second = x.itv[0].second;
    itv[1].first = x.itv[1].first;
    itv[1].second = x.itv[1].second;
    itv[2].first = x.itv[2].first;
    itv[2].second = x.itv[2].second;
        level=x.level;
        return *this;
    }

};

// the initialized error input, solve tolerance, time interval upper bound, etc.
class CCDConfig{
public:
    Scalar err_in[3];// the input error bound calculate from the AABB of the whole mesh
    Scalar co_domain_tolerance; // tolerance of the co-domain
    Scalar max_t; // the upper bound of the time interval
    int max_itr;// the maximal nbr of iterations
    

};

// the output info
class CCDOut{
public:
    bool result;
    Scalar toi=SCALAR_LIMIT;
    Scalar output_tolerance;
    int overflow_flag=NO_OVERFLOW;
    Scalar tol[3];// conservative domain tolerance
    //Scalar dbg[8];
};

// this is to record the interval related info
class BoxCompute{
public:
    item current_item;// containing 3 intervals and the level
    Scalar err[3]; // the error bound
    bool box_in=true; // if the inclusion function is inside the error bound
    Scalar true_tol=0; // the actual solving tolerance of the co-domain
    Scalar widths[3];
    int split;
};

// this is to calculate the vertices of the inclusion function
class BoxPrimatives{
public:
    bool b[3];
    int dim;
    Scalar t;
    Scalar u;
    Scalar v;
__device__ void calculate_tuv(const BoxCompute& box);
};
