#pragma once

#include<gputi/queue.h>

// this is the test for CCD. in the future this can be rewrite as an object of a single CCD query
class temp_body{
public:
    temp_body();

    // this is to test the gpu heap
    __device__ void insert_one_value(const item &ivalue);
    MinHeap heap;
    int size_of_heap();
 
};


