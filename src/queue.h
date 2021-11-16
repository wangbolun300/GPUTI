#pragma once
#include <gputi/Type.hpp>

namespace ccd{
// A class for Min Heap
class MinHeap
{
	int harr[HEAP_SIZE]; // pointer to array of elements in heap
	int capacity; // maximum possible size of min heap
	int heap_size; // Current number of elements in min heap
	int size_ever;
public:
	item Ilist[HEAP_SIZE]; // this list contains the interval3d
   __host__ __device__ MinHeap();
	// to heapify a subtree with the root at given index
	__host__ __device__ void MinHeapify();
	__host__ __device__ bool empty();
	// index
	__host__ __device__ int parent(int i) { return (i - 1) / 2; }

	// to get index of left child of node at index i
	__host__ __device__ int left(int i) { return (2 * i + 1); }

	// to get index of right child of node at index i
	__host__ __device__ int right(int i) { return (2 * i + 2); }

	// to extract the root which is the minimum element
	__device__ void extractMin(item &k);
	__host__ __device__ bool custom_compare_no_larger(const int &i1, const int &i2);
	__host__ __device__ bool custom_compare_less(const int &i1, const int &i2);
	// Decreases key value of key at index i to new_val
	// __host__ __device__ void decreaseKey(int i, item new_val);

	// Deletes a key stored at index i
	// __host__ __device__ void deleteKey(int i);

	// Inserts a new key 'k'
	__host__ __device__ bool insertKey(const item &k);
	__host__ __device__ bool insertKey(const Singleinterval si[3], const int &lv);
	__device__ void extractMinID(int&id);
	__device__ void SortAfterExtractMinID();
};

__device__ void split_dimension(const CCDOut& out,BoxCompute& box);

// t1+t2<=1? 
// true, when t1 + t2 < 1 / (1 + DBL_EPSILON);
// false, when  t1 + t2 > 1 / (1 - DBL_EPSILON);
// unknow, otherwise. 
__device__ bool sum_no_larger_1(const Scalar &num1, const Scalar &num2);

__device__ void compute_face_vertex_tolerance(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);
__device__ void compute_edge_edge_tolerance(const CCDdata &data_in,const CCDConfig& config, CCDOut& out);
__device__ __host__ void get_numerical_error_vf(
    const CCDdata &data_in,
    BoxCompute &box);
__device__ __host__ void get_numerical_error_ee(
    const CCDdata &data_in,
    BoxCompute &box);

	__device__ Scalar calculate_vf(const CCDdata &data_in, const BoxPrimatives& bp);
	__device__ Scalar calculate_ee(const CCDdata &data_in, const BoxPrimatives& bp);
	__device__ void split_dimension(const CCDOut& out,BoxCompute& box);
__device__ void bisect_vf_and_push(BoxCompute& box,const CCDConfig& config, MinHeap& istack,CCDOut& out);
__device__ void bisect_ee_and_push(BoxCompute& box,const CCDConfig& config, MinHeap& istack,CCDOut& out);
__host__ __device__ void item_equal(item& a, const item&b);
}