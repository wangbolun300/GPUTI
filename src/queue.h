#pragma once
#include <gputi/Type.hpp>

namespace ccd{
// A class for Min Heap
class MinHeap
{
	int harr[HEAP_SIZE]; // pointer to array of elements in heap
	int capacity; // maximum possible size of min heap
	int heap_size; // Current number of elements in min heap
public:
	item Ilist[HEAP_SIZE]; // this list contains the interval3d
   __device__ MinHeap();
	// to heapify a subtree with the root at given index
	__device__ void MinHeapify();
	__device__ bool empty();
	// index
	__device__ int parent(int i) { return (i - 1) / 2; }

	// to get index of left child of node at index i
	__device__ int left(int i) { return (2 * i + 1); }

	// to get index of right child of node at index i
	__device__ int right(int i) { return (2 * i + 2); }

	// to extract the root which is the minimum element
	__device__ void extractMin(item &k);
	__device__ bool custom_compare_no_larger(const int &i1, const int &i2);
	__device__ bool custom_compare_less(const int &i1, const int &i2);
	// Decreases key value of key at index i to new_val
	// __device__ void decreaseKey(int i, item new_val);

	// Deletes a key stored at index i
	// __device__ void deleteKey(int i);

	// Inserts a new key 'k'
	__device__ bool insertKey(const item &k);
	__device__ bool insertKey(const Singleinterval si[3], const int &lv);
};

__device__ void split_dimension(const CCDOut& out,BoxCompute& box);
}