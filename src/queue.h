#pragma once
#include <gputi/Type.hpp>
__device__ long power(const long a, const int b);
__device__ bool less_than(const Numccd &num1, const Numccd &num2);


// A class for Min Heap
class MinHeap
{
	item harr[HEAP_SIZE]; // pointer to array of elements in heap
	int capacity; // maximum possible size of min heap
	int heap_size; // Current number of elements in min heap
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
	// __device__ void decreaseKey(int i, item new_val);

	// Returns the minimum key (key at root) from min heap
	__device__ item getMin() { return harr[0]; }

	// Deletes a key stored at index i
	// __device__ void deleteKey(int i);

	// Inserts a new key 'k'
	__device__ bool insertKey(item k);
};