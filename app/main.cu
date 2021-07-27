#include<gputi/queue.h>
#include<iostream>

// A C++ program to demonstrate common Binary Heap Operations
#include<iostream>
#include<climits>

__device__ void get_array(MinHeap heap, int n, ptest& out){
  
    for(int i=0;i<n;i++){
        out[i]=heap.extractMin().key;
    }
}
__global__ void test_heap(ptest *ids, int m, int n, ptest *out){
    int tx=threadIdx.x;
    if (tx < m) {
        MinHeap heap;
       for(int i=0;i<n;i++){
           item it(ids[tx][i]);
           heap.insertKey(it);
       }
       for(int i=0;i<n;i++){
        //    get_array(heap,n,out[tx]);
          out[tx][i]=heap.extractMin().key;
       }
    }
}


void test_heap(){
    const int m=1024;// nbr of heaps
    const int n=5;// size of each heap
    ptest a[m];
    ptest out[m];
    srand(5);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            a[i][j]=rand();
        }
    }
    ptest *d_a,*d_out;
    int size = m*sizeof(ptest);
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_out,size);
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);

    test_heap<<<1,m>>>(d_a,m,n,d_out);
    cudaMemcpy(out,d_out,size,cudaMemcpyDeviceToHost);
    // for(int i=0;i<m;i++){
    //     for(int j=0;j<n;j++){
    //         std::cout<<a[i][j]<<", ";
    //     }
    //     std::cout<<std::endl;
    // }
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            std::cout<<out[i][j]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"the total size is "<<size<<" the size of int is "<<sizeof(int)<<std::endl;

}


int main(int argc, char ** argv){
    test_heap();
    // MinHeap h;
    //MinHeap h;

    // h.insertKey(item(3));
	// h.insertKey(item(2));
	// h.deleteKey(1);
	// h.insertKey(item(15));
	// h.insertKey(item(5));
	// h.insertKey(item(4));
	// h.insertKey(item(45));
    // std::cout << h.extractMin().key << " ";
    // std::cout << h.getMin().key << " ";
    // h.decreaseKey(2, 1);
    // std::cout << h.getMin().key;

    //std::cout<<"size "<<sizeof(MinHeap)<<std::endl;
    std::cout<<"done!"<<std::endl;
    return 1;
}
