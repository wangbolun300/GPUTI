#include<gputi/queue.h>
#include<iostream>
#include <limits>

using namespace std;


__device__ void interval_cp(const Singleinterval& a,Singleinterval& b){
	b.first.first=a.first.first;
	b.first.second=a.first.second;
	b.second.first=a.second.first;
	b.second.second=a.second.second;
} 
__device__ item::item(const Singleinterval* si, int lv) {
	level = lv;
	interval_cp(si[0],itv[0]);
	interval_cp(si[1],itv[1]);
	interval_cp(si[2],itv[2]);
	//itv[0].first.first=si[0].first.first;
	// itv[1]=si[1];
	// itv[2]=si[2];
}
__device__ item::item() {

}

__device__ item MinHeap::item_max() {
	
	it.level = INT_MAX;
	return it;
}
__device__ item MinHeap::item_min() {
	
	it.level = INT_MIN;
	return it;
}

// i1==i2?
__device__ bool custom_compare_equal(const item &i1, const item &i2) {
	if (i1.level == i2.level){
		if(i1.itv[0].first == i2.itv[0].first){
			return true;
		}
	}
	return false;
}


// i1<i2?
__device__ bool custom_compare_less(const item &i1, const item &i2) {
	if (i1.level != i2.level)
    { 
        if(i1.level < i2.level){
			return true;
		}
		else{
			return false;
		}
    }
	else{
		
        if(less_than(i1.itv[0].first, i2.itv[0].first)){
			return true;
		}
		else{
			return false;
		}
	}
	return true;
}

// i1<=i2?
__device__ bool custom_compare_no_larger(const item &i1, const item &i2) {
	
	return !custom_compare_less(i2,i1);;
}





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
__device__ bool MinHeap::insertKey(item k)
{	// to avoid overflow, instead of comparing with capacity, we compare with capacity -1
	if (heap_size == capacity-1)
	{
		//cout << "\nOverflow: Could not insertKey\n";
		return 0;
	}

	// First insert the new key at the end
	heap_size++;
	iki = heap_size - 1;
	harr[iki] = k;

	// Fix the min heap property if it is violated
	while (iki != 0 && !custom_compare_no_larger(harr[parent(iki)], harr[iki]))
	{
		swap(&harr[iki], &harr[parent(iki)]);
		iki = parent(iki);
	}
	return true;
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
	return item_max();
	// Store the minimum value, and remove it from heap
	root = harr[0];
	
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
// bolun remove the recursive part and make it a iteration
__device__ void MinHeap::MinHeapify(int i)
{
	
	tmp=i;
	
	while(1){
		l = left(tmp);
		r = right(tmp);
		smallest = tmp;
		if (l < heap_size && custom_compare_less(harr[l], harr[tmp]))
			smallest = l;
		if (r < heap_size && custom_compare_less(harr[r], harr[smallest]))
			smallest = r;
		if(smallest==tmp){
			return;
		}
		else{
			swap(&harr[tmp], &harr[smallest]);
			tmp=smallest;
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
__device__ bool  MinHeap::empty(){
	return (heap_size==0);
}

// A utility function to swap two elements
__device__ void MinHeap::swap(item *x, item *y)
{
	temp = *x;
	*x = *y;
	*y = temp;
}