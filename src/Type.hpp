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
    __device__ __host__ VectorMax3d(const Scalar &a, const Scalar &b, const Scalar &c);
   __device__ __host__ void init(const Scalar &a, const Scalar &b, const Scalar &c);
    Scalar v[3];

    
// __device__  __host__  friend VectorMax3d operator+(const VectorMax3d &x, const VectorMax3d &y)
//         {
//             VectorMax3d out;
//             out.v[0]=x.v[0]+y.v[0];
//             out.v[1]=x.v[1]+y.v[1];
//             out.v[2]=x.v[2]+y.v[2];
//             return out;
//         }
// __device__  __host__  friend VectorMax3d operator-(const VectorMax3d &x, const VectorMax3d &y)
//         {
//             VectorMax3d out;
//             out.v[0]=x.v[0]-y.v[0];
//             out.v[1]=x.v[1]-y.v[1];
//             out.v[2]=x.v[2]-y.v[2];
//             return out;
//         }
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
// __device__ void VecSum(const VectorMax3d& a, const VectorMax3d& b, const VectorMax3d& res);

// __device__ void VecMinus(const VectorMax3d& a, const VectorMax3d& b, const VectorMax3d& res);
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
class item {
public:
	int level;
	Singleinterval itv[3];
	__device__ item(const Singleinterval* si, int level);
	__device__ item();
	__device__ item& operator=(const item& x)
    {
        if (this == &x)
            return *this;
        itv[0]=x.itv[0];
		itv[1]=x.itv[1];
		itv[2]=x.itv[2];
        
        return *this;
    }

};

// A class for Min Heap
class MinHeap
{
	item harr[HEAP_SIZE]; // pointer to array of elements in heap
	int capacity; // maximum possible size of min heap
	int heap_size; // Current number of elements in min heap
	item root;
	int tmp;
	int l;
	int r;
	int smallest;
	item it;
	int iki;
	__device__ item item_max();
	__device__ item item_min();
	item temp;
	// Prototype of a utility function to swap two integers
	__device__ void swap(item *x, item *y);
public:
	// Constructor
	//MinHeap(int capacity);
   __device__ MinHeap();
	// to heapify a subtree with the root at given index
	__device__ void MinHeapify(int);
	__device__ bool empty();
	// index
	__device__ int parent(int i) { return (i - 1) / 2; }

	// to get index of left child of node at index i
	__device__ int left(int i) { return (2 * i + 1); }

	// to get index of right child of node at index i
	__device__ int right(int i) { return (2 * i + 2); }

	// to extract the root which is the minimum element
	__device__ item extractMin();

	// Decreases key value of key at index i to new_val
	__device__ void decreaseKey(int i, item new_val);

	// Returns the minimum key (key at root) from min heap
	__device__ item getMin() { return harr[0]; }

	// Deletes a key stored at index i
	__device__ void deleteKey(int i);

	// Inserts a new key 'k'
	__device__ bool insertKey(item k);
	
};
CCDdata array_to_ccd(std::array<std::array<Scalar,3>,8> a, bool is_edge);
__device__ void single_test_wrapper(CCDdata* vfdata, bool &result);
void print_vector(Scalar* v, int size);
void print_vector(int* v, int size);

// for interval b = a;
__device__ void interval_cp(const Singleinterval& a,Singleinterval& b);

class RFclass{
public:
__device__ bool interval_root_finder_double_horizontal_tree(
    const Scalar tol[3],
    const Scalar co_domain_tolerance,
    const Singleinterval iset[3],
    const bool check_t_overlap,
    const Scalar
        max_t, // check interval [0, max_t] when check_t_overlap is set as TRUE
    Scalar &toi,
    const bool check_vf,
    const Scalar err[3],
    const Scalar ms,
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    const int max_itr,
    Scalar &output_tolerance,
    int &overflow_flag);
Scalar temp_output_tolerance;
// current intervals
Singleinterval current[3];
Scalar true_tol[3];
int refine;
Scalar impact_ratio;
Scalar temp_toi;
Numccd TOI_SKIP;
Numccd TOI;
bool use_skip;
int current_level;
bool zero_in;
bool box_in;
int level;
item current_item;
VectorMax3d widths;
bool tol_condition;
bool condition1;
bool condition2;
bool condition3;
bool check[3];
VectorMax3d widthratio;
Scalar t_upper_bound;
bool this_level_less_tol;
int box_in_level;
bool find_level_root;
int split_i;
MinHeap istack;
interval_pair halves;
bool inserted;
};

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
Scalar tmp_opt;
};

class RFvar{
public:
bool check_t_overlap;
Numccd low_number;
Numccd up_number;
Singleinterval init_interval;
Singleinterval iset[3];
bool result;
RFclass rf;
};
class CCDvar{
public:
bool res;
const int MAX_NO_ZERO_TOI_ITER = SCALAR_LIMIT;
unsigned int no_zero_toi_iter;
bool is_impacting;
bool tmp_is_impacting;
Scalar tolerance_in;
Scalar ms_in;
Scalar tol[3];
Scalar err1[3];
VectorMax3d tol_v;
VectorMax3d vlist[8];
bool use_ms;
int itr;

ERRvar errvar;
RFvar rfvar;

};


