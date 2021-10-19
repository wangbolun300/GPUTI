#include <gputi/Type.hpp>
#include <iostream>
#include <limits>

using namespace std;

__device__ item::item(const Singleinterval si[3], const int &lv)
{
	level = lv;
	itv[0].first=si[0].first;
	itv[0].second=si[0].second;
	itv[1].first=si[1].first;
	itv[1].second=si[1].second;
	itv[2].first=si[2].first;
	itv[2].second=si[2].second;
}


// i1<i2?
__device__ bool custom_compare_less(const item &i1, const item &i2)
{
	if (i1.level < i2.level){
		return true;
	}
	if(i1.level == i2.level&&i1.itv[0].first< i2.itv[0].first){
		return true;
	}

	return false;
}




__device__ __host__ MinHeap::MinHeap()
{
	heap_size = 1;
	capacity = HEAP_SIZE;
	harr[0].itv[0].first = 0;
	harr[0].itv[0].second = 1;

	harr[0].itv[1].first = 0;
	harr[0].itv[1].second = 1;
	
	harr[0].itv[2].first = 0;
	harr[0].itv[2].second = 1;

	harr[0].level = -1;
}
__device__ void MinHeap::initialize()
{
	heap_size = 1;
	capacity = HEAP_SIZE;
	harr[0].itv[0].first = 0;
	harr[0].itv[0].second = 1;

	harr[0].itv[1].first = 0;
	harr[0].itv[1].second = 1;
	
	harr[0].itv[2].first = 0;
	harr[0].itv[2].second = 1;

	harr[0].level = -1;
}

// Inserts a new key 'k'
__device__ bool MinHeap::insertKey(const item &k)
{ // to avoid overflow, instead of comparing with capacity, we compare with capacity -1
	if (heap_size == capacity - 1)
	{
		return false;
	}

	// First insert the new key at the end

	itr = heap_size;

	harr[itr].itv[0].first = k.itv[0].first;
	harr[itr].itv[0].second = k.itv[0].second;

	harr[itr].itv[1].first = k.itv[1].first;
	harr[itr].itv[1].second = k.itv[1].second;

	harr[itr].itv[2].first = k.itv[2].first;
	harr[itr].itv[2].second = k.itv[2].second;
	
	harr[itr].level = k.level;
	heap_size++;

	// Fix the min heap property if it is violated
	status=custom_compare_less(harr[itr], harr[parent(itr)]);
	while (itr != 0 && status)
	{
		swap(&harr[itr], &harr[parent(itr)]);
		itr = parent(itr);
	}
	return true;
}

// Method to remove minimum element (or root) from min heap
__device__ item MinHeap::extractMin()
{
	// using our algorithm the heap size will be checked by is_empty()
	// if (heap_size <= 0)
	// 	return item_max();

	// Store the minimum value, and remove it from heap
	
	// un-nesting the operator "=" seems not work.
	root.itv[0].first = harr[0].itv[0].first;
    root.itv[0].second = harr[0].itv[0].second;
    root.itv[1].first = harr[0].itv[1].first;
    root.itv[1].second = harr[0].itv[1].second;
    root.itv[2].first = harr[0].itv[2].first;
    root.itv[2].second = harr[0].itv[2].second;
    root.level=harr[0].level;
	//root = harr[0];

	harr[0].itv[0].first = harr[heap_size - 1].itv[0].first;
    harr[0].itv[0].second = harr[heap_size - 1].itv[0].second;
    harr[0].itv[1].first = harr[heap_size - 1].itv[1].first;
    harr[0].itv[1].second = harr[heap_size - 1].itv[1].second;
    harr[0].itv[2].first = harr[heap_size - 1].itv[2].first;
    harr[0].itv[2].second = harr[heap_size - 1].itv[2].second;
    harr[0].level=harr[heap_size - 1].level;

	//harr[0] = harr[heap_size - 1];
	heap_size--;

	MinHeapify();

	return root;
}

__device__ void MinHeap::MinHeapify()
{
	tmp = 0;

	for (itr = 0;; itr++)
	{
		l = left(tmp);
		r = right(tmp);
		smallest = tmp;
		status=custom_compare_less(harr[l], harr[tmp]);
		if (l < heap_size && status)
			smallest = l;
		status=custom_compare_less(harr[r], harr[smallest]);
		if (r < heap_size && status)
			smallest = r;
		if (smallest == tmp)
		{
			return;
		}
		else
		{
			swap(&harr[tmp], &harr[smallest]);
			tmp = smallest;
		}
	}
}
__device__ bool MinHeap::empty()
{
	return (heap_size == 0);
}

// A utility function to swap two elements
__device__ void MinHeap::swap(item *x, item *y)
{
	
	itm = x;
	x = y;
	y = itm;
}
