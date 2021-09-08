#include <gputi/queue.h>
#include <iostream>
#include <limits>

using namespace std;

__device__ void interval_cp(const Singleinterval &a, Singleinterval &b)
{
	b.first.first = a.first.first;
	b.first.second = a.first.second;
	b.second.first = a.second.first;
	b.second.second = a.second.second;
}
__device__ item::item(const Singleinterval si[3], int lv)
{
	level = lv;
	interval_cp(si[0], itv[0]);
	interval_cp(si[1], itv[1]);
	interval_cp(si[2], itv[2]);
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
	bool ir = custom_compare_less(i2, i1);
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
		swap(harr[i], harr[parent(i)]);
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
	item_equal(temp, x);
	item_equal(x, y);
	item_equal(y, temp);
}