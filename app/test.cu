#include <gputi/root_finder.h>
#include <gputi/book.h>
#include "timer.hpp"
#include <iostream>
#include <functional>
#include <fstream>
#include "read_rational_csv.hpp"
#include <filesystem>
#include <cuda/std/functional>

#include <gputi/timer.cuh>

__device__ void interval_cp(const Singleinterval& a, Singleinterval& b)
{
	b.first.first = a.first.first;
	b.first.second = a.first.second;
	b.second.first = a.second.first;
	b.second.second = a.second.second;
}
__device__ item::item(const Singleinterval si[3], int lv)
{
	level = lv;
	recordLaunch<const Singleinterval &, Singleinterval &>("interval_cp_si0",  interval_cp, si[0], itv[0]);
    recordLaunch<const Singleinterval &, Singleinterval &>("interval_cp_si1",  interval_cp, si[1], itv[1]);
    recordLaunch<const Singleinterval &, Singleinterval &>("interval_cp_si2",  interval_cp, si[2], itv[2]);
    // interval_cp(si[0], itv[0]);
	// interval_cp(si[1], itv[1]);
	// interval_cp(si[2], itv[2]);
}
__device__ item::item()
{
}

__device__ item item_max()
{
	item it;
	it.level = INT_MAX;
	return it;
}
__device__ item item_min()
{
	item it;
	it.level = INT_MIN;
	return it;
}

// i1==i2?
__device__ bool custom_compare_equal(const item &i1, const item &i2)
{

	bool con1=i1.level == i2.level;
	bool con2=i1.itv[0].first == i2.itv[0].first;
	bool res=con1&&con2;
	return res;
}

// i1<i2?
__device__ bool custom_compare_less(const item &i1, const item &i2)
{
	if (i1.level != i2.level)
	{
		if (i1.level < i2.level)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{

		if (less_than(i1.itv[0].first, i2.itv[0].first))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	return true;
}

// i1<=i2?
__device__ bool custom_compare_no_larger(const item &i1, const item &i2)
{
	bool ir = recordLaunch<bool, const item &, const item &>("custom_compare_less", custom_compare_less, i2, i1);
	return !ir;
}

// Prototype of a utility function to swap two integers
__device__ void swap(item &x, item &y);

// Constructor: Builds a heap from a given array a[] of given size
// MinHeap::MinHeap(int cap)
// {
// 	heap_size = 0;
// 	capacity = cap;
// 	harr = new item[cap];
// }
__device__ MinHeap::MinHeap()
{
	heap_size = 0;
	capacity = HEAP_SIZE;
}

// Inserts a new key 'k'
__device__ bool MinHeap::insertKey(item k)
{ // to avoid overflow, instead of comparing with capacity, we compare with capacity -1
	if (heap_size == capacity - 1)
	{
		return false;
	}

	// First insert the new key at the end
	heap_size++;
	int i = heap_size - 1;
	item_equal(harr[i], k);

	// Fix the min heap property if it is violated
	while (i != 0 && !custom_compare_no_larger(harr[parent(i)], harr[i]))
	{
		recordLaunch<item&, item&>("swap", swap, harr[i], harr[parent(i)]);
		i = parent(i);
	}
	return true;
}

// Method to remove minimum element (or root) from min heap
__device__ item MinHeap::extractMin()
{

	if (heap_size <= 0)
		return recordLaunch("item_max", item_max);

	// Store the minimum value, and remove it from heap
	item root;

	root = harr[0];

	harr[0] = harr[heap_size - 1];
	heap_size--;
	// recordLaunch("MinHeapify", &MinHeap::MinHeapify);
    // std::function<void()> foo = [&]__device__(){this->MinHeapify();};
    // recordLaunch("MinHeapify", foo);
    clock_t start = clock();
    MinHeapify();
    clock_t stop = clock();
    double t = (double)stop - start;
    if (threadIdx.x+blockIdx.x == 0)
        printf ("%s: %d clicks (%f ms).\n", "MinHeapify", t,((float)t)/CLOCKS_PER_SEC*1000); 

	return root;
}

// This function deletes key at index i. It first reduced value to minus
// infinite, then calls extractMin()
// __device__ void MinHeap::deleteKey(int i)
// {
// 	decreaseKey(i, item_min());
// 	extractMin();
// }

// A recursive method to heapify a subtree with the root at given index
// This method assumes that the subtrees are already heapified
// bolun remove the recursive part and make it a iteration
__device__ void MinHeap::MinHeapify()
{
	int tmp = 0;
	//return;
	for (int itr = 0;; itr++)
	{
		int l = left(tmp);
		int r = right(tmp);
		int smallest = tmp;
		if (l < heap_size && custom_compare_less(harr[l], harr[tmp]))
			smallest = l;
		if (r < heap_size && custom_compare_less(harr[r], harr[smallest]))
			smallest = r;
		if (smallest == tmp)
		{
			return;
		}
		else
		{
			swap(harr[tmp], harr[smallest]);
			tmp = smallest;
		}
	}

	// int l = left(i);
	// int r = right(i);
	// int smallest = i;
	// if (l < heap_size && custom_compare_less(harr[l], harr[i]))
	// 	smallest = l;
	// if (r < heap_size && custom_compare_less(harr[r], harr[smallest]))
	// 	smallest = r;

	// if (smallest != i)
	// {
	// 	swap(&harr[i], &harr[smallest]);
	// 	MinHeapify(smallest);
	// }
}
__device__ bool MinHeap::empty()
{
	return (heap_size == 0);
}

// A utility function to swap two elements
__device__ void swap(item &x, item &y)
{
	item temp;
	recordLaunch<item&,const item&>("item_equal", item_equal, temp, x);
	recordLaunch<item&,const item&>("item_equal", item_equal, x, y);
	item_equal(y, temp);
}
__device__ void item_equal(item &a, const item &b)
{
    a.itv[0].first.first = b.itv[0].first.first;
    a.itv[0].first.second = b.itv[0].first.second;
    a.itv[0].second.first = b.itv[0].second.first;
    a.itv[0].second.second = b.itv[0].second.second;

    a.itv[1].first.first = b.itv[1].first.first;
    a.itv[1].first.second = b.itv[1].first.second;
    a.itv[1].second.first = b.itv[1].second.first;
    a.itv[1].second.second = b.itv[1].second.second;

    a.itv[2].first.first = b.itv[2].first.first;
    a.itv[2].first.second = b.itv[2].first.second;
    a.itv[2].second.first = b.itv[2].second.first;
    a.itv[2].second.second = b.itv[2].second.second;
    a.level = b.level;
}
void print_vector(Scalar *v, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << v[i] << ",";
    }
    std::cout << std::endl;
}
void print_vector(int *v, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << v[i] << ",";
    }
    std::cout << std::endl;
}

CCDdata array_to_ccd(std::array<std::array<Scalar, 3>, 8> a, bool is_edge)
{
    CCDdata data;
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        data.v0s[i] = a[0][i];
        data.v1s[i] = a[1][i];
        data.v2s[i] = a[2][i];
        data.v3s[i] = a[3][i];
        data.v0e[i] = a[4][i];
        data.v1e[i] = a[5][i];
        data.v2e[i] = a[6][i];
        data.v3e[i] = a[7][i];
    }
    data.is_edge = is_edge;
    return data;
}
__device__ __host__ VectorMax3d::VectorMax3d(Scalar a, Scalar b, Scalar c)
{
    v[0] = a;
    v[1] = b;
    v[2] = c;
}

__device__ long reduction(const long n, long &result)
{
    int t = 0;
    int newn = n;
    while (newn % 2 == 0)
    {
        newn = newn / 2;
        t++;
    }
    result = newn;
    return t;
}

__device__ Singleinterval::Singleinterval(Numccd f, Numccd s)
{
    first = f;
    second = s;
}
__device__ interval_pair::interval_pair(const Singleinterval &a, const Singleinterval &b)
{
    // interval_cp(a,first);
    // interval_cp(b,second);
    first = a;
    second = b;
}

//TODO add overflow checks for these basis functions
// calculate a*(2^b)
__device__ long power(const long a, const int b) { return a << b; }
__device__ Scalar Numccd2double(const Numccd &n)
{
    Scalar r = Scalar(n.first) / power(1, n.second);
    return r;
}
__device__ Numccd::Numccd(const long a, const int b)
{
    first = a;
    second = b;
}
__device__ bool sum_no_larger_1(const Numccd &num1, const Numccd &num2)
{
    long k1 = num1.first;
    int n1 = num1.second;
    long k2 = num2.first;
    int n2 = num2.second;
    long k;
    int n;
    if (n1 == n2)
    {
        k = k1 + k2;
        n = n1;
    }
    if (n1 < n2)
    {
        k = power(1, n2 - n1) * k1 + k2;
        n = n2;
    }
    if (n1 > n2)
    {
        k = power(1, n1 - n2) * k2 + k1;
        n = n1;
    }
    if (k > power(1, n))
        return false;
    else
        return true;
}
__device__ bool less_than(const Numccd &num1, const Numccd &num2)
{
    long k1 = num1.first;
    int n1 = num1.second;
    long k2 = num2.first;
    int n2 = num2.second;

    if (n1 < n2)
    {
        k1 = power(1, n2 - n1) * k1;
    }
    if (n1 > n2)
    {
        k2 = power(1, n1 - n2) * k2;
    }
    if (k1 < k2)
        return true;
    return false;
}
__device__ bool interval_overlap_region(
    const Singleinterval &itv, const Scalar r1, const Scalar r2)
{
    Scalar b1 = Numccd2double(itv.first);
    Scalar b2 = Numccd2double(itv.second);
    if (b2 < r1 || b1 > r2)
        return false;
    return true;
}

__device__ VectorMax3d width(const Singleinterval *x)
{
    VectorMax3d w;
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        w.v[i] = // 0.1;
            (Numccd2double(x[i].second) - Numccd2double(x[i].first));
    }
    return w;
}

__device__ void convert_tuv_to_array(
    const Singleinterval &itv0, const Singleinterval &itv1, Singleinterval &itv2,
    Scalar t_up[8],
    Scalar t_dw[8],
    Scalar u_up[8],
    Scalar u_dw[8],
    Scalar v_up[8],
    Scalar v_dw[8])
{
    // t order: 0,0,0,0,1,1,1,1
    // u order: 0,0,1,1,0,0,1,1
    // v order: 0,1,0,1,0,1,0,1
    Scalar t0_up = itv0.first.first,
           t0_dw = power(1, itv0.first.second),
           t1_up = itv0.second.first,
           t1_dw = power(1, itv0.second.second),

           u0_up = itv1.first.first,
           u0_dw = power(1, itv1.first.second),
           u1_up = itv1.second.first,
           u1_dw = power(1, itv1.second.second),

           v0_up = itv2.first.first,
           v0_dw = power(1, itv2.first.second),
           v1_up = itv2.second.first,
           v1_dw = power(1, itv2.second.second);

    t_up[0] = t0_up;
    t_up[1] = t0_up;
    t_up[2] = t0_up;
    t_up[3] = t0_up;
    t_up[4] = t1_up;
    t_up[5] = t1_up;
    t_up[6] = t1_up;
    t_up[7] = t1_up;
    t_dw[0] = t0_dw;
    t_dw[1] = t0_dw;
    t_dw[2] = t0_dw;
    t_dw[3] = t0_dw;
    t_dw[4] = t1_dw;
    t_dw[5] = t1_dw;
    t_dw[6] = t1_dw;
    t_dw[7] = t1_dw;
    u_up[0] = u0_up;
    u_up[1] = u0_up;
    u_up[2] = u1_up;
    u_up[3] = u1_up;
    u_up[4] = u0_up;
    u_up[5] = u0_up;
    u_up[6] = u1_up;
    u_up[7] = u1_up;
    u_dw[0] = u0_dw;
    u_dw[1] = u0_dw;
    u_dw[2] = u1_dw;
    u_dw[3] = u1_dw;
    u_dw[4] = u0_dw;
    u_dw[5] = u0_dw;
    u_dw[6] = u1_dw;
    u_dw[7] = u1_dw;
    v_up[0] = v0_up;
    v_up[1] = v1_up;
    v_up[2] = v0_up;
    v_up[3] = v1_up;
    v_up[4] = v0_up;
    v_up[5] = v1_up;
    v_up[6] = v0_up;
    v_up[7] = v1_up;
    v_dw[0] = v0_dw;
    v_dw[1] = v1_dw;
    v_dw[2] = v0_dw;
    v_dw[3] = v1_dw;
    v_dw[4] = v0_dw;
    v_dw[5] = v1_dw;
    v_dw[6] = v0_dw;
    v_dw[7] = v1_dw;
}

__device__ void function_vf(
    const Scalar &vs,
    const Scalar &t0s,
    const Scalar &t1s,
    const Scalar &t2s,
    const Scalar &ve,
    const Scalar &t0e,
    const Scalar &t1e,
    const Scalar &t2e,
    const Scalar t_up[8],
    const Scalar t_dw[8],
    const Scalar u_up[8],
    const Scalar u_dw[8],
    const Scalar v_up[8],
    const Scalar v_dw[8],
    Scalar rst[8])
{
    Scalar v ;
Scalar t0;
Scalar t1;
Scalar t2;
Scalar pt;
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
         v = (ve - vs) * t_up[i] / t_dw[i] + vs;
         t0 = (t0e - t0s) * t_up[i] / t_dw[i] + t0s;
         t1 = (t1e - t1s) * t_up[i] / t_dw[i] + t1s;
         t2 = (t2e - t2s) * t_up[i] / t_dw[i] + t2s;
         pt = (t1 - t0) * u_up[i] / u_dw[i] + (t2 - t0) * v_up[i] / v_dw[i] + t0;
        rst[i] = v - pt;
    }
}

__device__ void function_ee(
    const Scalar &a0s,
    const Scalar &a1s,
    const Scalar &b0s,
    const Scalar &b1s,
    const Scalar &a0e,
    const Scalar &a1e,
    const Scalar &b0e,
    const Scalar &b1e,
    const Scalar t_up[8],
    const Scalar t_dw[8],
    const Scalar u_up[8],
    const Scalar u_dw[8],
    const Scalar v_up[8],
    const Scalar v_dw[8],
    Scalar rst[8])
{
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        Scalar edge0_vertex0 = (a0e - a0s) * t_up[i] / t_dw[i] + a0s;
        Scalar edge0_vertex1 = (a1e - a1s) * t_up[i] / t_dw[i] + a1s;
        Scalar edge1_vertex0 = (b0e - b0s) * t_up[i] / t_dw[i] + b0s;
        Scalar edge1_vertex1 = (b1e - b1s) * t_up[i] / t_dw[i] + b1s;

        Scalar edge0_vertex =
            (edge0_vertex1 - edge0_vertex0) * u_up[i] / u_dw[i] + edge0_vertex0;
        Scalar edge1_vertex =
            (edge1_vertex1 - edge1_vertex0) * v_up[i] / v_dw[i] + edge1_vertex0;
        rst[i] = edge0_vertex - edge1_vertex;
    }
}

// ** this version can return the true x or y or z tolerance of the co-domain **
// eps is the interval [-eps,eps] we need to check
// if [-eps,eps] overlap, return true
// bbox_in_eps tell us if the box is totally in eps box
// ms is the minimum seperation
__device__ void evaluate_bbox_one_dimension_vector_return_tolerance(
    Scalar t_up[8],
    Scalar t_dw[8],
    Scalar u_up[8],
    Scalar u_dw[8],
    Scalar v_up[8],
    Scalar v_dw[8],
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    int dimension,
    const bool& check_vf,
    const Scalar& eps,
    const Scalar& ms,
    bool &bbox_in_eps,
    Scalar &tol,
    bool &result)
{
    
    Scalar vs[8];
    bbox_in_eps = false;

#ifndef CHECK_EE
    function_vf(
        a0s[dimension], a1s[dimension], b0s[dimension], b1s[dimension],
        a0e[dimension], a1e[dimension], b0e[dimension], b1e[dimension],
        t_up, t_dw, u_up, u_dw, v_up, v_dw, vs);
#else
    function_ee(
        a0s[dimension], a1s[dimension], b0s[dimension], b1s[dimension],
        a0e[dimension], a1e[dimension], b0e[dimension], b1e[dimension],
        t_up, t_dw, u_up, u_dw, v_up, v_dw, vs);
#endif

    Scalar minv = vs[0];
    Scalar maxv = vs[0];
    minv=fminf(vs[1],minv);
    maxv=fmaxf(vs[1],maxv);
    minv=fminf(vs[2],minv);
    maxv=fmaxf(vs[2],maxv);
    minv=fminf(vs[3],minv);
    maxv=fmaxf(vs[3],maxv);
    minv=fminf(vs[4],minv);
    maxv=fmaxf(vs[4],maxv);
    minv=fminf(vs[5],minv);
    maxv=fmaxf(vs[5],maxv);
    minv=fminf(vs[6],minv);
    maxv=fmaxf(vs[6],maxv);
    minv=fminf(vs[7],minv);
    maxv=fmaxf(vs[7],maxv);
// #pragma unroll
//     for (int i = 1; i < 8; i++)
//     {
//         if (minv > vs[i])
//         {
//             minv = vs[i];
//         }
//         if (maxv < vs[i])
//         {
//             maxv = vs[i];
//         }
//     }

    tol = maxv - minv; // this is the real tolerance
    
    if (minv - ms > eps || maxv + ms < -eps){
        result= false;
        return;
    }
       
    if (minv + ms >= -eps && maxv - ms <= eps)
    {
        bbox_in_eps = true;
    }
    result= true;
    return;
}

// we used to use this function, but now we directly do what this function does in the root-finder function
__device__ void Origin_in_function_bounding_box_double_vector_return_tolerance(
    Singleinterval paras[3],
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    const bool check_vf,
    const Scalar box[3],
    const Scalar ms,
    bool &box_in_eps,
    Scalar tolerance[3], bool &result)
{
    
    box_in_eps = false;
    result=false;
    Scalar t_up[8];
    Scalar t_dw[8];
    Scalar u_up[8];
    Scalar u_dw[8];
    Scalar v_up[8];
    Scalar v_dw[8];
    Singleinterval itv0 = paras[0], itv1 = paras[1], itv2 = paras[2];

    recordLaunch<const Singleinterval &, const Singleinterval &, Singleinterval &,
    Scalar *,
    Scalar *,
    Scalar *,
    Scalar *,
    Scalar *,
    Scalar *>("convert_tuv_to_array", convert_tuv_to_array, itv0, itv1, itv2, t_up, t_dw, u_up, u_dw, v_up, v_dw);
    
    //bool ck;
    bool box_in[3];

// #pragma unroll
//     for (int i = 0; i < 3; i++)
//     {

//         ck = evaluate_bbox_one_dimension_vector_return_tolerance(
//             t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
//             a1e, b0e, b1e, i, check_vf, box[i], ms, box_in[i],
//             tolerance[i]);

//         if (!ck)
//         {
//             return false;
//         }
//     }
    // if (box_in[0] && box_in[1] && box_in[2])
    // {
    //     box_in_eps = true;
    // }
    bool ck0,ck1,ck2;
    recordLaunch<Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], int, const bool &, const Scalar &, const Scalar &, bool &, Scalar &, bool &>("evaluate_bbox_one_dimension_vector_return_tolerance_t0", evaluate_bbox_one_dimension_vector_return_tolerance,
        t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
        a1e, b0e, b1e, 0, check_vf, box[0], ms, box_in[0],
        tolerance[0],ck0);
    recordLaunch<Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], int, const bool &, const Scalar &, const Scalar &, bool &, Scalar &, bool &>("evaluate_bbox_one_dimension_vector_return_tolerance_t1", evaluate_bbox_one_dimension_vector_return_tolerance, 
        t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
        a1e, b0e, b1e, 1, check_vf, box[1], ms, box_in[1],
        tolerance[1],ck1);
    recordLaunch<Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], int, const bool &, const Scalar &, const Scalar &, bool &, Scalar &, bool &>("evaluate_bbox_one_dimension_vector_return_tolerance_t2", evaluate_bbox_one_dimension_vector_return_tolerance, 
        t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
        a1e, b0e, b1e, 2, check_vf, box[2], ms, box_in[2],
        tolerance[2],ck2);
    
    box_in_eps = box_in[0] * box_in[1] * box_in[2];
    //int aa=int(ck0)*int(ck1)*int(ck2);
    // return;
    if(ck0&&ck1&&ck2){
        result=true;
    }
    
    
    
    // if (!ck)
    // {
    //     return false;
    // }
    
    // return ck;
    return;
}

__device__ void bisect(const Singleinterval &inter, interval_pair &out)
{
    Numccd low = inter.first;
    Numccd up = inter.second;

    // interval is [k1/pow(2,n1), k2/pow(2,n2)], k1,k2,n1,n2 are all not negative
    long k1 = low.first;
    int n1 = low.second;
    long k2 = up.first;
    int n2 = up.second;

    long k;
    int n;
    int p;
    long r;
    if (n2 == n1)
    {
        p = reduction(k1 + k2, r);
        k = r;
        n = n2 - p + 1;
    }
    if (n2 > n1)
    {
        k = k1 * power(1, n2 - n1) + k2;
        n = n2 + 1;
    }
    if (n2 < n1)
    {
        k = k1 + k2 * power(1, n1 - n2);
        n = n1 + 1;
    }
    Numccd newnum(k, n);
    Singleinterval i1(low, newnum), i2(newnum, up);
    out.first = i1;
    out.second = i2;
}

__device__ bool interval_root_finder_double_horizontal_tree(
    const Scalar tol[3],
    const Scalar &co_domain_tolerance,
    const Singleinterval iset[3],
    const bool &check_t_overlap,
    const Scalar
        &max_t, // check interval [0, max_t] when check_t_overlap is set as TRUE
    Scalar &toi,
    const bool &check_vf,
    const Scalar err[3],
    const Scalar &ms,
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    const int &max_itr,
    Scalar &output_tolerance,
    int &overflow_flag)
{
    
    overflow_flag = NO_OVERFLOW;
    // if max_itr <0, output_tolerance= co_domain_tolerance;
    // else, output_tolearance will be the precision after iteration time > max_itr
    output_tolerance = co_domain_tolerance;

    // this is used to catch the tolerance for each level
    Scalar temp_output_tolerance = co_domain_tolerance;

    // current intervals
    Singleinterval current[3];
    Scalar true_tol[3];
    int refine = 0;

    toi = SCALAR_LIMIT; //set toi as infinate
    // temp_toi is to catch the toi of each level
    Scalar temp_toi = toi;
    Numccd TOI;
    TOI.first = 4;
    TOI.second = 0; // set TOI as 4. this is to record the impact time of this level
    Numccd TOI_SKIP =
        TOI;               // this is to record the element that already small enough or contained in eps-box
    bool use_skip = false; // this is to record if TOI_SKIP is used.

    
    int current_level = -2; // in the begining, current_level != level
    int box_in_level = -2;  // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
    bool this_level_less_tol = true;
    bool find_level_root = false;
    // Scalar current_tolerance=std::numeric_limits<Scalar>::infinity(); // set returned tolerance as infinite
    Scalar t_upper_bound = max_t; // 2*tol make it more conservative

    
    MinHeap istack;
    
    bool zero_in;
    bool box_in;
    Scalar t_up[8];
    Scalar t_dw[8];
    Scalar u_up[8];
    Scalar u_dw[8];
    Scalar v_up[8];
    Scalar v_dw[8];
    int level;
    bool box_in_[3];
    bool ck0,ck1,ck2;
    Singleinterval itv0, itv1, itv2;
 
    istack.insertKey(item(iset, -1));
    item current_item;
    while (!istack.empty())
    {
        if(overflow_flag!=NO_OVERFLOW){
            break;
        }
        current_item = istack.extractMin();

        current[0] = current_item.itv[0];
        current[1] = current_item.itv[1];
        current[2] = current_item.itv[2];
        level = current_item.level;

        // if this box is later than TOI_SKIP in time, we can skip this one.
        // TOI_SKIP is only updated when the box is small enough or totally contained in eps-box
        if (!less_than(current[0].first, TOI_SKIP))
        {
            continue;
        }
        if (box_in_level != level)
        { // before check a new level, set this_level_less_tol=true
            box_in_level = level;
            this_level_less_tol = true;
        }

        refine++;
    
    box_in = false;
    zero_in=false;
    
    itv0 = current[0]; itv1 = current[1]; itv2 = current[2];

    convert_tuv_to_array(itv0, itv1, itv2, t_up, t_dw, u_up, u_dw, v_up, v_dw);
    
    //bool ck;

// zero_in=true;
// box_in=true;
// box_in_[0]=false;
// box_in_[1]=false;
// box_in_[2]=false;
//     #pragma unroll
//     for (int i = 0; i < 3; i++)
//     {

// evaluate_bbox_one_dimension_vector_return_tolerance(
//         t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
//         a1e, b0e, b1e, i, check_vf, err[i], ms, box_in_[i],
//         true_tol[i],ck);

//         if (!ck)
//         {
//             zero_in=false;
//             break;
//         }
//     }
    recordLaunch<Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], int, const bool &, const Scalar &, const Scalar &, bool &, Scalar &, bool &>("evaluate_bbox_one_dimension_vector_return_tolerance_t0", evaluate_bbox_one_dimension_vector_return_tolerance, 
        t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
        a1e, b0e, b1e, 0, check_vf, err[0], ms, box_in_[0],
        true_tol[0],ck0);
    recordLaunch<Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], int, const bool &, const Scalar &, const Scalar &, bool &, Scalar &, bool &>("evaluate_bbox_one_dimension_vector_return_tolerance_t1", evaluate_bbox_one_dimension_vector_return_tolerance, 
        t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
        a1e, b0e, b1e, 1, check_vf, err[1], ms, box_in_[1],
        true_tol[1],ck1);
    recordLaunch<Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], Scalar [8], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], const Scalar [3], int, const bool &, const Scalar &, const Scalar &, bool &, Scalar &, bool &>("evaluate_bbox_one_dimension_vector_return_tolerance_t2", evaluate_bbox_one_dimension_vector_return_tolerance, 
        t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e,
        a1e, b0e, b1e, 2, check_vf, err[2], ms, box_in_[2],
        true_tol[2],ck2);
    
    box_in = box_in_[0] && box_in_[1] && box_in_[2];
    zero_in=ck0&&ck1&&ck2;
    

        
            // Origin_in_function_bounding_box_double_vector_return_tolerance(
            //     current, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, check_vf,
            //     err, ms, box_in, true_tol,zero_in);

        if (!zero_in)
            continue;
        
        VectorMax3d widths = recordLaunch<VectorMax3d, const Singleinterval *>("width", width, current);

        bool tol_condition = true_tol[0] <= co_domain_tolerance && true_tol[1] <= co_domain_tolerance && true_tol[2] <= co_domain_tolerance;

        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
        bool condition1 = widths.v[0] <= tol[0] && widths.v[1] <= tol[1] && widths.v[2] <= tol[2];

        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        bool condition2 = box_in && this_level_less_tol;
        if (!tol_condition)
        {
            this_level_less_tol = false;
            // this level has at least one box whose size > tolerance, thus we
            // cannot directly return if find one box whose size < tolerance or box-in
        }

        // Condition 3, in this level, we find a box that zero-in and size < tolerance.
        // and no other boxes whose zero-in is true in this level before this one is larger than tolerance, can return
        bool condition3 = this_level_less_tol;
        if (condition1 || condition2 || condition3)
        {
            TOI = current[0].first;
            
            // continue;
            toi = Numccd2double(TOI);

            return true;

        }

       
            if (current_level != level)
            {
                current_level = level;
                find_level_root = false;
            }
            if (!find_level_root)
            {
                TOI = current[0].first;

                
                // continue;
                temp_toi = Numccd2double(TOI);

                // if the real tolerance is larger than input, use the real one;
                // if the real tolerance is smaller than input, use input
                temp_output_tolerance = max(
                    max(
                        max(true_tol[0], true_tol[1]), true_tol[2]),
                    co_domain_tolerance);

                find_level_root =
                    true; // this ensures always find the earlist root
            }
            if (refine > max_itr)
            {
                overflow_flag = ITERATION_OVERFLOW;
                break;
            }
        

        // if this box is small enough, or inside of eps-box, then just continue,
        // but we need to record the collision time
        if (tol_condition || box_in)
        {
            if (less_than(current[0].first, TOI_SKIP))
            {
                TOI_SKIP = current[0].first;
            }
            use_skip = true;
            continue;
        }

        bool check[3];
        VectorMax3d widthratio;

        check[0] = false;
        check[1] = false;
        check[2] = false;
        for (int i = 0; i < 3; i++)
        {
            widthratio.v[i] = widths.v[i] / tol[i];
            if (widths.v[i] > tol[i])
                check[i] = true; // means this need to be checked
        }

        int split_i = -1;

        for (int i = 0; i < 3; i++)
        {
            if (check[i])
            {
                if (check[(i + 1) % 3] && check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 1) % 3] && widthratio.v[i] >= widthratio.v[(i + 2) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (check[(i + 1) % 3] && !check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 1) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (!check[(i + 1) % 3] && check[(i + 2) % 3])
                {
                    if (widthratio.v[i] >= widthratio.v[(i + 2) % 3])
                    {
                        split_i = i;
                        break;
                    }
                }
                if (!check[(i + 1) % 3] && !check[(i + 2) % 3])
                {

                    split_i = i;
                    break;
                }
            }
        }

        interval_pair halves;
        Singleinterval bisect_inter = current[split_i];

        bisect(bisect_inter, halves);
        if (!less_than(halves.first.first, halves.first.second))
        {
            overflow_flag = BISECTION_OVERFLOW;
            break;
        }
        if (!less_than(halves.second.first, halves.second.second))
        {
            overflow_flag = BISECTION_OVERFLOW;
            break;
        }
#ifndef CHECK_EE // check vf

        if (split_i == 1)
        {

            if (recordLaunch<bool, const Numccd &, const Numccd &>("sum_no_larger_1-halves.second", sum_no_larger_1, halves.second.first, current[2].first))
            {

                current[split_i] = halves.second;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (recordLaunch<bool, const Numccd &, const Numccd &>("sum_no_larger_1-halves.first", sum_no_larger_1, halves.first.first, current[2].first))
            {
                current[split_i] = halves.first;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }

        if (split_i == 2)
        {

            if (recordLaunch<bool, const Numccd &, const Numccd &>("sum_no_larger_1-halves.second", sum_no_larger_1, halves.second.first, current[1].first))
            {
                current[split_i] = halves.second;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (recordLaunch<bool, const Numccd &, const Numccd &>("sum_no_larger_1-halves.first",sum_no_larger_1, halves.first.first, current[1].first))
            {
                current[split_i] = halves.first;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }
        if (split_i == 0)
        {
            if (check_t_overlap)
            {
                if (recordLaunch<bool, const Singleinterval &, Scalar, Scalar>("interval_overlap_region-halves.second", interval_overlap_region,
                        halves.second, 0, t_upper_bound))
                {
                    current[split_i] = halves.second;
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
                if (recordLaunch<bool, const Singleinterval &, Scalar, Scalar>("interval_overlap_region-halves.first", interval_overlap_region,
                        halves.first, 0, t_upper_bound))
                {
                    current[split_i] = halves.first;
                    bool inserted = istack.insertKey(item(current, level + 1));
                    if (inserted == false)
                    {
                        overflow_flag = HEAP_OVERFLOW;
                    }
                }
            }
            else
            {
                current[split_i] = halves.second;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
                current[split_i] = halves.first;
                inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }

#else
        if (check_t_overlap && split_i == 0)
        {
            if (recordLaunch<bool, const Singleinterval &, Scalar, Scalar>("interval_overlap_region-halves.second", interval_overlap_region, 
                    halves.second, 0, t_upper_bound))
            {
                current[split_i] = halves.second;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
            if (recordLaunch<bool, const Singleinterval &, Scalar, Scalar>("interval_overlap_region-halves.first", interval_overlap_region, halves.first, 0, t_upper_bound))
            {
                current[split_i] = halves.first;
                bool inserted = istack.insertKey(item(current, level + 1));
                if (inserted == false)
                {
                    overflow_flag = HEAP_OVERFLOW;
                }
            }
        }
        else
        {
            current[split_i] = halves.second;
            bool inserted = istack.insertKey(item(current, level + 1));
            if (inserted == false)
            {
                overflow_flag = HEAP_OVERFLOW;
            }
            current[split_i] = halves.first;
            inserted = istack.insertKey(item(current, level + 1));
            if (inserted == false)
            {
                overflow_flag = HEAP_OVERFLOW;
            }
        }
#endif
    }

    if (overflow_flag > 0)
    {
        toi = temp_toi;
        output_tolerance = temp_output_tolerance;
        return true;
    }

    if (use_skip)
    {
        toi = Numccd2double(TOI_SKIP);

        return true;
    }

    return false;
}

__device__ bool interval_root_finder_double_horizontal_tree(
    const Scalar tol[3],
    const Scalar &co_domain_tolerance,
    Scalar &toi,
    const bool check_vf,
    const Scalar err[3], // this is the maximum error on each axis when calculating the vertices, err, aka, filter
    const Scalar ms,
    const Scalar a0s[3],
    const Scalar a1s[3],
    const Scalar b0s[3],
    const Scalar b1s[3],
    const Scalar a0e[3],
    const Scalar a1e[3],
    const Scalar b0e[3],
    const Scalar b1e[3],
    const Scalar max_time,
    const int max_itr,
    Scalar &output_tolerance,
    int &overflow_flag)
{
#ifdef TIME_UPPER_IS_ONE
    bool check_t_overlap = false; // if input max_time = 1, then no need to check overlap
#else
    bool check_t_overlap = true;
#endif

const Numccd low_number= Numccd(0,0);
const Numccd up_number=Numccd(1,0);
const Singleinterval init_interval= Singleinterval(low_number, up_number);
    Singleinterval iset[3];
    iset[0] = init_interval;
    iset[1] = init_interval;
    iset[2] = init_interval;

    // bool result = recordLaunch<bool, const Scalar *, const Scalar, Singleinterval [3], bool, const Scalar, Scalar, const bool, const Scalar *, const Scalar, const Scalar *, const Scalar *, const Scalar *, const Scalar *, const Scalar *, const Scalar *, const Scalar *, const Scalar *, const int, Scalar, int>("interval_root_finder_double_horizontal_tree", interval_root_finder_double_horizontal_tree,
    bool result =   interval_root_finder_double_horizontal_tree(  
    tol, co_domain_tolerance, iset, check_t_overlap, max_time, toi,
        check_vf, err, ms, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, max_itr,
        output_tolerance, overflow_flag);

    return result;
}

__device__ Scalar max_linf_dist(const VectorMax3d &p1, const VectorMax3d &p2)
{
    Scalar r = 0;
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        if (r < fabs(p1.v[i] - p2.v[i]))
        {
            r = fabs(p1.v[i] - p2.v[i]);
        }
    }
    return r;
}

__device__ Scalar max_linf_4(
    const VectorMax3d &p1,
    const VectorMax3d &p2,
    const VectorMax3d &p3,
    const VectorMax3d &p4,
    const VectorMax3d &p1e,
    const VectorMax3d &p2e,
    const VectorMax3d &p3e,
    const VectorMax3d &p4e)
{
    Scalar r = 0, temp = 0;
    temp = recordLaunch<Scalar, const VectorMax3d &, const VectorMax3d &>("max_linf_dist-p1", max_linf_dist, p1e, p1);
    if (r < temp)
        r = temp;
    temp = recordLaunch<Scalar, const VectorMax3d &, const VectorMax3d &>("max_linf_dist-p2", max_linf_dist, p2e, p2);
    if (r < temp)
        r = temp;
    temp = recordLaunch<Scalar, const VectorMax3d &, const VectorMax3d &>("max_linf_dist-p3", max_linf_dist, p3e, p3);
    if (r < temp)
        r = temp;
    temp = recordLaunch<Scalar, const VectorMax3d &, const VectorMax3d &>("max_linf_dist-p4", max_linf_dist, p4e, p4);
    if (r < temp)
        r = temp;
    return r;
}
__device__ void compute_face_vertex_tolerance_3d_new(
    const CCDdata &data_in,
    const Scalar tolerance, Scalar *result)
{
    VectorMax3d vs(data_in.v0s[0], data_in.v0s[1], data_in.v0s[2]);
    VectorMax3d f0s(data_in.v1s[0], data_in.v1s[1], data_in.v1s[2]);
    VectorMax3d f1s(data_in.v2s[0], data_in.v2s[1], data_in.v2s[2]);
    VectorMax3d f2s(data_in.v3s[0], data_in.v3s[1], data_in.v3s[2]);
    VectorMax3d ve(data_in.v0e[0], data_in.v0e[1], data_in.v0e[2]);
    VectorMax3d f0e(data_in.v1e[0], data_in.v1e[1], data_in.v1e[2]);
    VectorMax3d f1e(data_in.v2e[0], data_in.v2e[1], data_in.v2e[2]);
    VectorMax3d f2e(data_in.v3e[0], data_in.v3e[1], data_in.v3e[2]);
    VectorMax3d p000 = vs - f0s, p001 = vs - f2s,
                p011 = vs - (f1s + f2s - f0s), p010 = vs - f1s;
    VectorMax3d p100 = ve - f0e, p101 = ve - f2e,
                p111 = ve - (f1e + f2e - f0e), p110 = ve - f1e;
    Scalar dl = 0;
    Scalar edge0_length = 0;
    Scalar edge1_length = 0;
    dl = 3 * recordLaunch<Scalar ,const VectorMax3d &, const VectorMax3d &, const VectorMax3d &, const VectorMax3d &, const VectorMax3d &, const VectorMax3d &, const VectorMax3d &, const VectorMax3d &>("max_linf_4", max_linf_4, p000, p001, p011, p010, p100, p101, p111, p110);
    edge0_length =
        3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    edge1_length =
        3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);

    result[0] = tolerance / dl;
    result[1] = tolerance / edge0_length;
    result[2] = tolerance / edge1_length;
}
__device__ void compute_edge_edge_tolerance_new(
    const CCDdata &data_in,
    const Scalar tolerance, Scalar *result)
{
    VectorMax3d edge0_vertex0_start(data_in.v0s[0], data_in.v0s[1], data_in.v0s[2]); // a0s
    VectorMax3d edge0_vertex1_start(data_in.v1s[0], data_in.v1s[1], data_in.v1s[2]); // a1s
    VectorMax3d edge1_vertex0_start(data_in.v2s[0], data_in.v2s[1], data_in.v2s[2]); // b0s
    VectorMax3d edge1_vertex1_start(data_in.v3s[0], data_in.v3s[1], data_in.v3s[2]); // b1s
    VectorMax3d edge0_vertex0_end(data_in.v0e[0], data_in.v0e[1], data_in.v0e[2]);
    VectorMax3d edge0_vertex1_end(data_in.v1e[0], data_in.v1e[1], data_in.v1e[2]);
    VectorMax3d edge1_vertex0_end(data_in.v2e[0], data_in.v2e[1], data_in.v2e[2]);
    VectorMax3d edge1_vertex1_end(data_in.v3e[0], data_in.v3e[1], data_in.v3e[2]);
    VectorMax3d p000 = edge0_vertex0_start - edge1_vertex0_start,
                p001 = edge0_vertex0_start - edge1_vertex1_start,
                p011 = edge0_vertex1_start - edge1_vertex1_start,
                p010 = edge0_vertex1_start - edge1_vertex0_start;
    VectorMax3d p100 = edge0_vertex0_end - edge1_vertex0_end,
                p101 = edge0_vertex0_end - edge1_vertex1_end,
                p111 = edge0_vertex1_end - edge1_vertex1_end,
                p110 = edge0_vertex1_end - edge1_vertex0_end;
    Scalar dl = 0;
    Scalar edge0_length = 0;
    Scalar edge1_length = 0;

    dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    edge0_length =
        3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    edge1_length =
        3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);
    result[0] = tolerance / dl;
    result[1] = tolerance / edge0_length;
    result[2] = tolerance / edge1_length;
}

__device__ __host__ void get_numerical_error(
    const VectorMax3d vertices[8], const int vsize,
    const bool &check_vf,
    const bool using_minimum_separation,
    Scalar *error)
{
    Scalar eefilter;
    Scalar vffilter;
#ifdef NO_CHECK_MS

#ifdef GPUTI_USE_DOUBLE_PRECISION
    eefilter = 6.217248937900877e-15;
    vffilter = 6.661338147750939e-15;
#else
    eefilter = 3.337861e-06;
    vffilter = 3.576279e-06;
#endif

#else
#ifdef GPUTI_USE_DOUBLE_PRECISION
    eefilter = 7.105427357601002e-15;
    vffilter = 7.549516567451064e-15;
#else
    eefilter = 3.814698e-06;
    vffilter = 4.053116e-06;
#endif

#endif

    Scalar xmax = fabs(vertices[0].v[0]);
    Scalar ymax = fabs(vertices[0].v[1]);
    Scalar zmax = fabs(vertices[0].v[2]);

    for (int i = 0; i < vsize; i++)
    {
        if (xmax < fabs(vertices[i].v[0]))
        {
            xmax = fabs(vertices[i].v[0]);
        }
        if (ymax < fabs(vertices[i].v[1]))
        {
            ymax = fabs(vertices[i].v[1]);
        }
        if (zmax < fabs(vertices[i].v[2]))
        {
            zmax = fabs(vertices[i].v[2]);
        }
    }
    Scalar delta_x = xmax > 1 ? xmax : 1;
    Scalar delta_y = ymax > 1 ? ymax : 1;
    Scalar delta_z = zmax > 1 ? zmax : 1;
#ifdef CHECK_EE
    error[0] = delta_x * delta_x * delta_x * eefilter;
    error[1] = delta_y * delta_y * delta_y * eefilter;
    error[2] = delta_z * delta_z * delta_z * eefilter;
#else
    error[0] = delta_x * delta_x * delta_x * vffilter;
    error[1] = delta_y * delta_y * delta_y * vffilter;
    error[2] = delta_z * delta_z * delta_z * vffilter;
#endif
    return;
}

__device__ bool CCD_Solver(
    const CCDdata &data_in,
    const Scalar err[3],
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag,
    bool is_vf)
{

    overflow_flag = 0;

    bool is_impacting;
    

    Scalar tol[3];
    // this is the error of the whole mesh
    Scalar err1[3];
#ifdef CHECK_EE
        recordLaunch<const CCDdata &, Scalar, Scalar *>("compute_edge_edge_tolerance_new", compute_edge_edge_tolerance_new, data_in, tolerance, tol);
#else
        recordLaunch<const CCDdata &, Scalar, Scalar *>("compute_face_vertex_tolerance_3d_new", compute_face_vertex_tolerance_3d_new, data_in, tolerance, tol);
#endif

        //////////////////////////////////////////////////////////

#ifdef CALCULATE_ERROR_BOUND
        VectorMax3d vlist[8];
#pragma unroll
        for (int i = 0; i < 3; i++)
        {
            vlist[0].v[i] = data_in.v0s[i];
            vlist[1].v[i] = data_in.v1s[i];
            vlist[2].v[i] = data_in.v2s[i];
            vlist[3].v[i] = data_in.v3s[i];
            vlist[4].v[i] = data_in.v0e[i];
            vlist[5].v[i] = data_in.v1e[i];
            vlist[6].v[i] = data_in.v2e[i];
            vlist[7].v[i] = data_in.v3e[i];
        }

        bool use_ms = ms > 0;
        recordLaunch<const VectorMax3d *, int, const bool &, bool, Scalar *>("get_numerical_error", get_numerical_error, vlist, 8, is_vf, use_ms, err1);
#else
        err1[0] = err[0];
        err1[1] = err[1];
        err1[2] = err[2];
#endif

#ifdef TIME_UPPER_IS_ONE
    bool check_t_overlap = false; // if input max_time = 1, then no need to check overlap
#else
    bool check_t_overlap = true;
#endif

    const Numccd low_number(0, 0);
    const Numccd up_number(1, 0);
    const Singleinterval init_interval(low_number, up_number);
    Singleinterval iset[3];
    iset[0] = init_interval;
    iset[1] = init_interval;
    iset[2] = init_interval;

    is_impacting = interval_root_finder_double_horizontal_tree(
    // is_impacting = recordLaunch("interval_root_finder_double_horizontal_tree", interval_root_finder_double_horizontal_tree,
        tol, tolerance, iset, check_t_overlap, t_max, toi,
        is_vf, err1, ms, data_in.v0s,data_in.v1s,data_in.v2s,data_in.v3s,
        data_in.v0e,data_in.v1e,data_in.v2e,data_in.v3e, max_itr,
        output_tolerance, overflow_flag);
    if (overflow_flag)
    {
        return true;
    }
    
    return is_impacting;
}

__device__ bool vertexFaceCCD_double(
    const CCDdata &data_in,
    const Scalar *err,
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag)
{
    
    bool res = recordLaunch<bool, const CCDdata &, const Scalar *, Scalar, Scalar &, Scalar, Scalar, int, Scalar &, bool, int &, bool>("CCD_Solver", CCD_Solver,data_in, err, ms, toi, tolerance, t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag, true);
    return res;
}

__device__ bool edgeEdgeCCD_double(
    const CCDdata &data_in,
    const Scalar *err,
    const Scalar ms,
    Scalar &toi,
    Scalar tolerance,
    Scalar t_max,
    const int max_itr,
    Scalar &output_tolerance,
    bool no_zero_toi,
    int &overflow_flag)
{
    bool res = CCD_Solver(data_in, err, ms, toi, tolerance, t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag, false);
    return res;
}

std::vector<std::string> simulation_folders = {{"chain", "cow-heads", "golf-ball", "mat-twist"}};
std::vector<std::string> handcrafted_folders = {{"erleben-sliding-spike", "erleben-spike-wedge",
                                                 "erleben-sliding-wedge", "erleben-wedge-crack", "erleben-spike-crack",
                                                 "erleben-wedges", "erleben-cube-cliff-edges", "erleben-spike-hole",
                                                 "erleben-cube-internal-edges", "erleben-spikes", "unit-tests"}};
struct Args
{
    std::string data_dir;
    double minimum_separation = 0;
    double tight_inclusion_tolerance = 1e-6;
    long tight_inclusion_max_iter = 1e6;
    bool run_ee_dataset = true;
    bool run_vf_dataset = true;
    bool run_simulation_dataset = true;
    bool run_handcrafted_dataset = true;
};
void print_V(std::array<std::array<Scalar, 3>, 8> V)
{
    for (int i = 0; i < 8; i++)
    {
        std::cout << V[i][0] << ", " << V[i][1] << ", " << V[i][2] << std::endl;
        if (i == 3)
        {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}
std::array<std::array<Scalar, 3>, 8> substract_ccd(const std::vector<std::array<Scalar, 3>> &data, int nbr)
{
    std::array<std::array<Scalar, 3>, 8> result;
    int start = nbr * 8;
    for (int i = 0; i < 8; i++)
    {
        result[i] = data[i + start];
    }
    return result;
}
void write_summary(
    const std::string file,
    const int method,
    const int total_number,
    const int positives,
    const bool is_edge_edge,
    const int fp,
    const int fn,
    const double average_time,
    const double time_lower,
    const double time_upper)
{
    std::ofstream fout;
    fout.open(file);
    fout << "method,total_nbr,positives,is_edge_edge,fp,fn,average_time, time_lower, time_upper"
         << std::endl;
    fout << method << "," << total_number << "," << positives << ","
         << is_edge_edge << "," << fp << "," << fn << ',' << average_time << "," << time_lower << "," << time_upper
         << std::endl;
    fout.close();
}
template <typename T>
void write_csv(const std::string &file, const std::vector<std::string> titles, const std::vector<T> data, bool large_info)
{
    std::cout << "inside write" << std::endl;
    std::ofstream fout;
    fout.open(file);

    if (large_info)
    {
        fout << "data" << std::endl;
        for (int i = 0; i < data.size(); i++)
        {
            fout << data[i] << std::endl;
        }
    }
    else
    {
        for (int i = 0; i < titles.size() - 1; i++)
        {
            fout << titles[i] << ",";
        }
        fout << titles.back() << std::endl;
        for (int i = 0; i < data.size() - 1; i++)
        {
            fout << data[i] << ",";
        }
        fout << data.back() << std::endl;
    }

    fout.close();
}
std::vector<std::string> file_path_base()
{
    // path is in the form of "chain/edge-edge/"
    std::vector<std::string> result;
    result.reserve(9999);
    for (int i = 1; i < 10000; i++)
    {
        std::string base;
        if (i < 10)
        {
            base = "000" + std::to_string(i);
        }
        if (i >= 10 && i < 100)
        {
            base = "00" + std::to_string(i);
        }
        if (i >= 100 && i < 1000)
        {
            base = "0" + std::to_string(i);
        }
        if (i >= 1000 && i < 10000)
        {
            base = std::to_string(i);
        }
        result.push_back(base);
    }
    return result;
}

__device__ void single_test_wrapper_return_toi(CCDdata *data, bool &result, Scalar &time_impact)
{
    
    Scalar err[3];
    err[0] = -1;
    err[1] = -1;
    err[2] = -1;
    Scalar ms = 0;
    Scalar toi;
    Scalar tolerance = 1e-6;
    Scalar t_max = 1;
    int max_itr = 1e6;
    Scalar output_tolerance;
    bool no_zero_toi = false;
    int overflow_flag;
    bool is_edge = data->is_edge;
    CCDdata data_cp;
    for (int i = 0; i < 3; i++)
    {
        data_cp.v0s[i] = data->v0s[i];
        data_cp.v1s[i] = data->v1s[i];
        data_cp.v2s[i] = data->v2s[i];
        data_cp.v3s[i] = data->v3s[i];
        data_cp.v0e[i] = data->v0e[i];
        data_cp.v1e[i] = data->v1e[i];
        data_cp.v2e[i] = data->v2e[i];
        data_cp.v3e[i] = data->v3e[i];
    }
    data_cp.is_edge = is_edge;
    
#ifdef CHECK_EE
        result = recordLaunch("edgeEdgeCCD_double", edgeEdgeCCD_double, data_cp, err, ms, toi, tolerance,
                              t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag);
#else
        result = recordLaunch<bool,const CCDdata &, const Scalar *, Scalar, Scalar &, Scalar, Scalar, int, Scalar &, bool, int &>("vertexFaceCCD_double", vertexFaceCCD_double,data_cp, err, ms, toi, tolerance,
                                      t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag);
#endif
    time_impact = toi;
    return;
}

__global__ void run_parallel_ccd_all(CCDdata *data, bool *res, int size, Scalar *tois)
{

    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tx < size)
    {
        CCDdata *input = &data[tx];
        bool result;
        Scalar toi;
        recordLaunch<CCDdata *, bool &, Scalar &>("single_test_wrapper_return_toi", single_test_wrapper_return_toi, input, result, toi);
        res[tx] = result;
        tois[tx] = toi;
    }
}



void all_ccd_run(const std::vector<std::array<std::array<Scalar, 3>, 8>> &V, bool is_edge,
                 std::vector<bool> &result_list, double &run_time, std::vector<Scalar> &time_impact, int parallel_nbr)
{
    int nbr = V.size();
    result_list.resize(nbr);
    // host
    CCDdata *data_list = new CCDdata[nbr];
    for (int i = 0; i < nbr; i++)
    {
        data_list[i] = array_to_ccd( V[i], is_edge);
    }
    bool *res = new bool[nbr];
    Scalar *tois = new Scalar[nbr];

    // device
    CCDdata *d_data_list;
    bool *d_res;
    Scalar *d_tois;

    int data_size = sizeof(CCDdata) * nbr;
    int result_size = sizeof(bool) * nbr;
    int time_size = sizeof(Scalar) * nbr;

    cudaMalloc(&d_data_list, data_size);
    cudaMalloc(&d_res, result_size);
    cudaMalloc(&d_tois, time_size);
    cudaMemcpy(d_data_list, data_list, data_size, cudaMemcpyHostToDevice);

    ccd::Timer timer;

    cudaProfilerStart();
    timer.start();
    run_parallel_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>(d_data_list, d_res, nbr, d_tois);
    cudaDeviceSynchronize();
    double tt = timer.getElapsedTimeInMicroSec();
    run_time = tt;
    cudaProfilerStop();

    cudaMemcpy(res, d_res, result_size, cudaMemcpyDeviceToHost);

    cudaMemcpy(tois, d_tois, time_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data_list);
    cudaFree(d_res);
    cudaFree(d_tois);

    for (int i = 0; i < nbr; i++)
    {
        result_list[i] = res[i];
    }

    time_impact.resize(nbr);

    for (int i = 0; i < nbr; i++)
    {
        time_impact[i] = tois[i];
    }

    delete[] res;
    delete[] data_list;
    delete[] tois;
    cudaError_t ct = cudaGetLastError();
    printf("******************\n%s\n************\n", cudaGetErrorString(ct));	
    return;
}


bool WRITE_STATISTIC = true;

void run_rational_data_single_method_parallel(
    const Args &args,
    const bool is_edge_edge,
    const bool is_simulation_data, int parallel, const std::string folder = "", const std::string tail = "")
{
    std::vector<std::array<Scalar, 3>> all_V;
    std::vector<bool> results;

    //std::vector<write_format> queryinfo;
    int total_number = -1;
    int total_positives = 0;
    int num_false_positives = 0;
    int num_false_negatives = 0;
    double time_lower = 1e100;
    double time_upper = -1;
    std::string sub_folder = is_edge_edge ? "/edge-edge/" : "/vertex-face/";
    std::string sub_name = is_edge_edge ? "edge-edge" : "vertex-face";
    std::vector<long> queue_sizes;
    std::vector<Scalar> tois;

    std::vector<bool> result_list;
    std::vector<bool> expect_list;
    std::vector<std::array<std::array<Scalar, 3>, 8>> queries;
    const std::vector<std::string> &scene_names = is_simulation_data ? simulation_folders : handcrafted_folders;
    std::cout << "loading data" << std::endl;
    std::vector<std::string> bases = file_path_base();
    for (const auto &scene_name : scene_names)
    {
        std::string scene_path = args.data_dir + scene_name + sub_folder;

        bool skip_folder = false;
        for (const auto &entry : bases)
        {
            if (skip_folder)
            {
                break;
            }
            std::string filename = scene_path + sub_name + "-" + entry + ".csv";

            // std::cout<<"filename "<<filename<<std::endl;
            // exit(0);
            if (queries.size() > TEST_NBR_QUERIES)
            {
                break;
            }
            all_V = ccd::read_rational_csv(filename, results);
            if (all_V.size() == 0)
            {
                std::cout << "data size " << all_V.size() << std::endl;
                std::cout << filename << std::endl;
            }

            if (all_V.size() == 0)
            {
                skip_folder = true;
                continue;
            }

            int v_size = all_V.size() / 8;
            for (int i = 0; i < v_size; i++)
            {
                if (queries.size() > TEST_NBR_QUERIES)
                {
                    break;
                }
                total_number += 1;

                std::array<std::array<Scalar, 3>, 8> V = substract_ccd(all_V, i);
                bool expected_result = results[i * 8];
                queries.push_back(V);
                expect_list.push_back(expected_result);
            }
        }
    }
    int size = queries.size();
    std::cout << "data loaded, size " << queries.size() << std::endl;
    double tavg = 0;
    int max_query_cp_size = 1e7;
    int start_id = 0;

    result_list.resize(size);
    tois.resize(size);

    while (1)
    {
        std::vector<bool> tmp_results;
        std::vector<std::array<std::array<Scalar, 3>, 8>> tmp_queries;
        std::vector<Scalar> tmp_tois;

        int remain = size - start_id;
        double tmp_tall;

        if (remain <= 0)
            break;

        int tmp_nbr = min(remain, max_query_cp_size);
        tmp_results.resize(tmp_nbr);
        tmp_queries.resize(tmp_nbr);
        tmp_tois.resize(tmp_nbr);
        for (int i = 0; i < tmp_nbr; i++)
        {
            tmp_queries[i] = queries[start_id + i];
        }
        all_ccd_run(tmp_queries, is_edge_edge, tmp_results, tmp_tall, tmp_tois, parallel);

        tavg += tmp_tall;
        for (int i = 0; i < tmp_nbr; i++)
        {
            result_list[start_id + i] = tmp_results[i];
            tois[start_id + i] = tmp_tois[i];
        }

        start_id += tmp_nbr;
    }
    tavg /= size;
    std::cout << "avg time " << tavg << std::endl;

    if (expect_list.size() != size)
    {
        std::cout << "size wrong!!!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < size; i++)
    {
        if (expect_list[i])
        {
            total_positives++;
        }
        if (result_list[i] != expect_list[i])
        {
            if (expect_list[i])
            {
                num_false_negatives++;
            }
            else
            {
                num_false_positives++;
            }
        }
    }
    std::cout << "total positives " << total_positives << std::endl;
    std::cout << "num_false_positives " << num_false_positives << std::endl;
    std::cout << "num_false_negatives " << num_false_negatives << std::endl;
    total_number = size;
    if (WRITE_STATISTIC)
    {
        write_summary(
            folder + "method" + std::to_string(int(2021)) + "_is_edge_edge_" + std::to_string(is_edge_edge) + "_" + std::to_string(total_number) + tail + ".csv",
            2021, total_number, total_positives, is_edge_edge,
            num_false_positives, num_false_negatives,
            tavg, time_lower, time_upper);
    }

    if (1)
    {
        std::vector<std::string> titles;
        write_csv(folder + "method" + std::to_string(int(2021)) + "_is_edge_edge_" + std::to_string(is_edge_edge) + "_" +
                      std::to_string(total_number) + "_tois" + tail + ".csv",
                  titles, tois, true);

        // write_csv(folder + "method" + std::to_string(int(2021)) + "_is_edge_edge_" + std::to_string(is_edge_edge) + "_" +
        // std::to_string(total_number) + "_runtime" + tail + ".csv", titles, time_list, true);
    }
}

void run_one_method_over_all_data(const Args &args, int parallel,
                                  const std::string folder = "", const std::string tail = "")
{
    if (args.run_handcrafted_dataset)
    {
        std::cout << "Running handcrafted dataset:\n";
        if (args.run_vf_dataset)
        {
            std::cout << "Vertex-Face:" << std::endl;
            run_rational_data_single_method_parallel(
                args, /*is_edge_edge=*/false, /*is_simu_data=*/false, parallel, folder, tail);
        }
        if (args.run_ee_dataset)
        {
            std::cout << "Edge-Edge:" << std::endl;
            run_rational_data_single_method_parallel(
                args, /*is_edge_edge=*/true, /*is_simu_data=*/false, parallel, folder, tail);
        }
    }
    if (args.run_simulation_dataset)
    {
        std::cout << "Running simulation dataset:\n";
        if (args.run_vf_dataset)
        {
            std::cout << "Vertex-Face:" << std::endl;
            run_rational_data_single_method_parallel(
                args, /*is_edge_edge=*/false, /*is_simu_data=*/true, parallel, folder, tail);
        }
        if (args.run_ee_dataset)
        {
            std::cout << "Edge-Edge:" << std::endl;
            run_rational_data_single_method_parallel(
                args, /*is_edge_edge=*/true, /*is_simu_data=*/true, parallel, folder, tail);
        }
    }
}
void run_ours_float_for_all_data(int parallel)
{
    std::string folder = std::string(getenv("HOME")) + "/data0809/"; // this is the output folder
    std::string tail = "_prl_" + std::to_string(parallel);
    Args arg;
    arg.data_dir = std::string(getenv("HOME")) + "/float_with_gt/";

    arg.minimum_separation = 0;
    arg.tight_inclusion_tolerance = 1e-6;
    arg.tight_inclusion_max_iter = 1e6;
    #ifdef CHECK_EE
    arg.run_ee_dataset = true;
    arg.run_vf_dataset = false;
    #else
    arg.run_ee_dataset = false;
    arg.run_vf_dataset = true;
    #endif
    arg.run_simulation_dataset = true;
    arg.run_handcrafted_dataset = false;
    run_one_method_over_all_data(arg, parallel, folder, tail);

}
int main(int argc, char **argv)
{
    // int deviceCount;
    //     cudaGetDeviceCount(&deviceCount);
    //     for(int i=0;i<deviceCount;i++)
    //     {
    //         cudaDeviceProp devProp;
    //         cudaGetDeviceProperties(&devProp, i);
    //         std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
    //         std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
    //         std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    //         std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    //         std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    //         std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
    //         std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    //         std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    //         std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
    //         std::cout << "======================================================" << std::endl;

    //     }
    //     return 0;
    // int alpha=5;
    // double* a = new double[alpha];
    int parallel = 0;
    if (argc == 1)
    {
        parallel = 1;
    }
    else
    {
        parallel = std::stoi(argv[1]);
    }
    if (parallel <= 0)
    {
        std::cout << "wrong parallel nbr = " << parallel << std::endl;
        return 0;
    }

    run_ours_float_for_all_data(parallel);
    std::cout << "done!" << std::endl;
    return 1;
}
