#include<gputi/queue.h>
#include<iostream>

using namespace std;
  

__device__ item::item(int i) {
	key = i;
}
__device__ item::item() {

}

__device__ item item_max() {
	item it;
	it.key = INT_MAX;
	return it;
}
__device__ item item_min() {
	item it;
	it.key = INT_MIN;
	return it;
}
__device__ bool custom_compare_no_larger(const item &a, const item &b) {
	if (a.key <= b.key) {
		return true;
	}
	return false;
}
__device__ bool custom_compare_less(const item &a, const item &b) {
	if (a.key < b.key) {
		return true;
	}
	return false;
}

// Prototype of a utility function to swap two integers
__device__ void swap(item *x, item *y);



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
	//harr = new item[cap];
}

// Inserts a new key 'k'
__device__ void MinHeap::insertKey(item k)
{	// to avoid overflow, instead of comparing with capacity, we compare with capacity -1
	if (heap_size == capacity-1)
	{
		//cout << "\nOverflow: Could not insertKey\n";
		return;
	}

	// First insert the new key at the end
	heap_size++;
	int i = heap_size - 1;
	harr[i] = k;

	// Fix the min heap property if it is violated
	while (i != 0 && !custom_compare_no_larger(harr[parent(i)], harr[i]))
	{
		swap(&harr[i], &harr[parent(i)]);
		i = parent(i);
	}
}

// Decreases value of key at index 'i' to new_val. It is assumed that
// new_val is smaller than harr[i].
__device__ void MinHeap::decreaseKey(int i, item new_val)
{
	harr[i] = new_val;
	while (i != 0 && !custom_compare_no_larger(harr[parent(i)], harr[i]))
	{
		swap(&harr[i], &harr[parent(i)]);
		i = parent(i);
	}
}

// Method to remove minimum element (or root) from min heap
__device__ item MinHeap::extractMin()
{
	if (heap_size <= 0)
		return item_max();
	if (heap_size == 1)
	{
		heap_size--;
		return harr[0];
	}

	// Store the minimum value, and remove it from heap
	item root = harr[0];
	harr[0] = harr[heap_size - 1];
	heap_size--;
	MinHeapify(0);

	return root;
}


// This function deletes key at index i. It first reduced value to minus
// infinite, then calls extractMin()
__device__ void MinHeap::deleteKey(int i)
{
	decreaseKey(i, item_min());
	extractMin();
}

// A recursive method to heapify a subtree with the root at given index
// This method assumes that the subtrees are already heapified
__device__ void MinHeap::MinHeapify(int i)
{
	int l = left(i);
	int r = right(i);
	int smallest = i;
	if (l < heap_size && custom_compare_less(harr[l], harr[i]))
		smallest = l;
	if (r < heap_size && custom_compare_less(harr[r], harr[smallest]))
		smallest = r;
	if (smallest != i)
	{
		swap(&harr[i], &harr[smallest]);
		MinHeapify(smallest);
	}
}

// A utility function to swap two elements
__device__ void swap(item *x, item *y)
{
	item temp = *x;
	*x = *y;
	*y = temp;
}