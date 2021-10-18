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

	bool con1 = i1.level == i2.level;
	bool con2 = i1.itv[0].first == i2.itv[0].first;
	return con1 && con2;
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

		if (i1.itv[0].first< i2.itv[0].first)
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
	return !custom_compare_less(i2, i1);
}

// Prototype of a utility function to swap two integers
__device__ void swap(item *x, item *y);

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

	int i = heap_size;

	harr[i].itv[0].first = k.itv[0].first;
	harr[i].itv[0].second = k.itv[0].second;

	harr[i].itv[1].first = k.itv[1].first;
	harr[i].itv[1].second = k.itv[1].second;

	harr[i].itv[2].first = k.itv[2].first;
	harr[i].itv[2].second = k.itv[2].second;
	
	harr[i].level = k.level;
	heap_size++;

	// Fix the min heap property if it is violated
	while (i != 0 && !custom_compare_no_larger(harr[parent(i)], harr[i]))
	{
		swap(&harr[i], &harr[parent(i)]);
		i = parent(i);
	}
	return true;
}

// Method to remove minimum element (or root) from min heap
__device__ item MinHeap::extractMin()
{

	if (heap_size <= 0)
		return item_max();

	// Store the minimum value, and remove it from heap
	item root;

	root = harr[0];

	harr[0] = harr[heap_size - 1];
	heap_size--;

	MinHeapify();

	return root;
}

__device__ void MinHeap::MinHeapify()
{
	int tmp = 0;
	int l;
	int r;
	int smallest;
	for (int itr = 0;; itr++)
	{
		l = left(tmp);
		r = right(tmp);
		smallest = tmp;
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
__device__ void swap(item *x, item *y)
{
	item *temp;
	temp = x;
	x = y;
	y = temp;
}
