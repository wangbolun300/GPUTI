#include<gputi/queue.h>
#include<iostream>

// A C++ program to demonstrate common Binary Heap Operations
#include<iostream>
#include<climits>
#include<gputi/root_finder.h>

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

__device__ void test_vector(const Scalar* v1, Scalar* v2){
    v2[0]=v1[0];
    v2[1]=v1[1];
    v2[2]=v1[2];
}
__global__ void test_vers(Scalar* a, Scalar* b){
    test_vector(a,b);
}

// __device__ void pass_value(Singleinterval* itv, Scalar* v0,Scalar* v1,Scalar* v2,Scalar* v3,Scalar* v4,Scalar* v5){
//     Singleinterval s[3];
//     s[0]=itv->[0];
// }
__device__ void check(Singleinterval *s, int* v){
    for(int i=0;i<8;i++){
        v[i]=1;//s->second.second;//1

    }
}
__global__ void test_cvt(Scalar* v0,Scalar* v1,Scalar* v2,Scalar* v3,Scalar* v4,Scalar* v5 , int *v6){
    Singleinterval itv[3];
    Numccd zero, one;
    zero.first=0;zero.second=0;
    one.first=1;one.second=0;
    Singleinterval s;
    s.first=zero;
    s.second=one;
    itv[0]=s;
    itv[1]=s;
    itv[2]=s;
    Singleinterval *i0, *i1, *i2;
    i0=new Singleinterval(zero,one);
    i1=new Singleinterval(zero,one);
    i2=new Singleinterval(zero,one);
    
    
    // v6[0]=1;
    // v6[1]=1;
    //check(i0,v6);
    convert_tuv_to_array(i0,i1,i2,v0,v1,v2,v3,v4,v5);
}

void print_vector(Scalar* v, int size){
    for(int i=0;i<size;i++){
        std::cout<<v[i]<<",";
    }
    std::cout<<std::endl;
}
void print_vector(int* v, int size){
    for(int i=0;i<size;i++){
        std::cout<<v[i]<<",";
    }
    std::cout<<std::endl;
}
void run_test(){
    Singleinterval itv[3];
    
    Scalar v0[8];Scalar v1[8];Scalar v2[8];Scalar v3[8];Scalar v4[8];Scalar v5[8]; int v6[8];
    Scalar* d_v0;Scalar* d_v1;Scalar* d_v2;Scalar* d_v3;Scalar* d_v4;Scalar* d_v5;
    int *d_v6;

    int vsize=sizeof(Scalar)*8;
    cudaMalloc(&d_v0,vsize);
    cudaMalloc(&d_v1,vsize);
    cudaMalloc(&d_v2,vsize);
    cudaMalloc(&d_v3,vsize);
    cudaMalloc(&d_v4,vsize);
    cudaMalloc(&d_v5,vsize);
    cudaMalloc(&d_v6,sizeof(int)*8);
    std::cout<<"before call function"<<std::endl;
    test_cvt<<<1,1>>>(d_v0,d_v1,d_v2,d_v3,d_v4,d_v5,d_v6);
 std::cout<<"after call function"<<std::endl;
    cudaMemcpy(v0,d_v0,vsize,cudaMemcpyDeviceToHost);
    cudaMemcpy(v1,d_v1,vsize,cudaMemcpyDeviceToHost);
    cudaMemcpy(v2,d_v2,vsize,cudaMemcpyDeviceToHost);
    cudaMemcpy(v3,d_v3,vsize,cudaMemcpyDeviceToHost);
    cudaMemcpy(v4,d_v4,vsize,cudaMemcpyDeviceToHost);
    cudaMemcpy(v5,d_v5,vsize,cudaMemcpyDeviceToHost);
    cudaMemcpy(v6,d_v6,sizeof(int)*8,cudaMemcpyDeviceToHost);
    std::cout<<"copied back"<<std::endl;
    print_vector(v0,8);
    print_vector(v1,8);
    print_vector(v2,8);
    print_vector(v3,8);
    print_vector(v4,8);
    print_vector(v5,8);
    print_vector(v6,8);
    std::cout<<v6[0]<<" "<<v6[1]<<std::endl;
    // Scalar a[3], b[3];
    // a[0]=1;
    // a[1]=2;
    // a[2]=3;
    // Scalar* ad,* bd;
    // int size=sizeof(Scalar)*3;
    // cudaMalloc(&ad,size);
    // cudaMalloc(&bd,size);
    // cudaMemcpy(ad,a,size,cudaMemcpyHostToDevice);

    // test_vers<<<1,1>>>(ad,bd);
    // cudaMemcpy(b,bd,size,cudaMemcpyDeviceToHost);
    // std::cout<<"out is "<<b[0]<<" "<<b[1]<<" "<<b[2]<<std::endl;
}
class tclass{

};
int main(int argc, char ** argv){
    run_test();
    //test_heap();
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
