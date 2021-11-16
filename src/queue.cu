#include <gputi/queue.h>
#include <iostream>
#include <limits>
namespace ccd{
using namespace std;

__host__ __device__ item::item(const Singleinterval si[3], const int &lv)
{
	level = lv;
	itv[0].first=si[0].first;
	itv[0].second=si[0].second;
	itv[1].first=si[1].first;
	itv[1].second=si[1].second;
	itv[2].first=si[2].first;
	itv[2].second=si[2].second;
}

// make a = b but avoid large registers usages
__host__ __device__ void item_equal(item& a, const item&b){
	a.level = b.level;
	a.itv[0].first=b.itv[0].first;
	a.itv[0].second=b.itv[0].second;
	a.itv[1].first=b.itv[1].first;
	a.itv[1].second=b.itv[1].second;
	a.itv[2].first=b.itv[2].first;
	a.itv[2].second=b.itv[2].second;
}
__host__ __device__ item::item()
{
}


// Ilist[i1] < Ilist[i2] ?
__host__ __device__ bool MinHeap::custom_compare_less(const int &i1, const int &i2)
{
	if (Ilist[i1].level != Ilist[i2].level)
	{
		return Ilist[i1].level < Ilist[i2].level;
	}
	else
	{
		return Ilist[i1].itv[0].first< Ilist[i2].itv[0].first;
	}
	return true;
}

// Ilist[i1] <= Ilist[i2] ?
__host__ __device__ bool MinHeap::custom_compare_no_larger(const int &i1, const int &i2)
{
	return !custom_compare_less(i2, i1);
}

// Prototype of a utility function to swap two integers
__host__ __device__ void swap(int &x, int &y);

__host__ __device__ MinHeap::MinHeap()
{
	heap_size = 1;
	capacity = HEAP_SIZE;
	Ilist[0].itv[0].first = 0;
	Ilist[0].itv[0].second = 1;

	Ilist[0].itv[1].first = 0;
	Ilist[0].itv[1].second = 1;
	
	Ilist[0].itv[2].first = 0;
	Ilist[0].itv[2].second = 1;

	Ilist[0].level = -1;
	// #pragma unroll
	// for(int i=0;i<HEAP_SIZE;i++){
	// 	harr[i]=i;
	// }
	harr[0]=0;
	size_ever=1;
}

// Inserts a new key 'k'
__host__ __device__ bool MinHeap::insertKey(const item &k)
{ // to avoid overflow, instead of comparing with capacity, we compare with capacity -1
	if (heap_size == capacity - 1)
	{
		return false;
	}

	// First insert the new key at the end


	int i = heap_size;
	if(i+1>size_ever){
		size_ever=i+1;
		harr[i]=i;
	}
	Ilist[harr[i]].itv[0].first = k.itv[0].first;
	Ilist[harr[i]].itv[0].second = k.itv[0].second;

	Ilist[harr[i]].itv[1].first = k.itv[1].first;
	Ilist[harr[i]].itv[1].second = k.itv[1].second;

	Ilist[harr[i]].itv[2].first = k.itv[2].first;
	Ilist[harr[i]].itv[2].second = k.itv[2].second;
	
	Ilist[harr[i]].level = k.level;
	heap_size++;

	// Fix the min heap property if it is violated
	while (i != 0 && !custom_compare_no_larger(harr[parent(i)], harr[i]))
	{
		swap(harr[i], harr[parent(i)]);
		i = parent(i);
	}
	return true;
}
__host__ __device__ bool MinHeap::insertKey(const Singleinterval si[3], const int &lv)
{
	if (heap_size == capacity - 1)
	{
		return false;
	}

	// First insert the new key at the end


	int i = heap_size;
	if(i+1>size_ever){
		size_ever=i+1;
		harr[i]=i;
	}

	Ilist[harr[i]].itv[0].first = si[0].first;
	Ilist[harr[i]].itv[0].second = si[0].second;

	Ilist[harr[i]].itv[1].first = si[1].first;
	Ilist[harr[i]].itv[1].second = si[1].second;

	Ilist[harr[i]].itv[2].first = si[2].first;
	Ilist[harr[i]].itv[2].second = si[2].second;
	
	Ilist[harr[i]].level =lv;
	heap_size++;

	// Fix the min heap property if it is violated
	while (i != 0 && !custom_compare_no_larger(harr[parent(i)], harr[i]))
	{
		swap(harr[i], harr[parent(i)]);
		i = parent(i);
	}
	return true;
}
// Method to remove minimum element (or root) from min heap
 __device__ void MinHeap::extractMin(item &k)
{
	// since our algorithm will detect if it is extractable, we will never return item_max()
	// if (heap_size <= 0)
	// 	return item_max();

	// Store the minimum value, and remove it from heap
	int root;

	root = harr[0];

	swap(harr[0], harr[heap_size - 1]);
	heap_size--;

	MinHeapify();

	k= Ilist[root];
	//item_equal(k,Ilist[root]);
}

__device__ void MinHeap::extractMinID(int&id){
	id=harr[0];
}

__device__ void MinHeap::SortAfterExtractMinID(){
	heap_size--;
	
	swap(harr[0], harr[heap_size]);
	

	MinHeapify();
}

__host__ __device__ void MinHeap::MinHeapify()
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
}
__host__ __device__ bool MinHeap::empty()
{
	return (heap_size == 0);
}

// A utility function to swap two elements
__host__ __device__ void swap(int &x, int &y)
{
	int temp = x;
    x = y;
    y = temp;
}
}