
#include<gputi/queue.h>
#include<iostream>

// A C++ program to demonstrate common Binary Heap Operations
#include<iostream>
#include<climits>
#include<gputi/root_finder.h>

// __device__ void get_array(MinHeap heap, int n, ptest& out){
  
//     for(int i=0;i<n;i++){
//         out[i]=heap.extractMin().key;
//     }
// }
// __global__ void test_heap(ptest *ids, int m, int n, ptest *out){
//     int tx=threadIdx.x;
//     if (tx < m) {
//         MinHeap heap;
//        for(int i=0;i<n;i++){
//            item it(ids[tx][i]);
//            heap.insertKey(it);
//        }
//        for(int i=0;i<n;i++){
//         //    get_array(heap,n,out[tx]);
//           out[tx][i]=heap.extractMin().key;
//        }
//     }
// }


// void test_heap(){
//     const int m=1024;// nbr of heaps
//     const int n=5;// size of each heap
//     ptest a[m];
//     ptest out[m];
//     srand(5);
//     for(int i=0;i<m;i++){
//         for(int j=0;j<n;j++){
//             a[i][j]=rand();
//         }
//     }
//     ptest *d_a,*d_out;
//     int size = m*sizeof(ptest);
//     cudaMalloc(&d_a,size);
//     cudaMalloc(&d_out,size);
//     cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);

//     test_heap<<<1,m>>>(d_a,m,n,d_out);
//     cudaMemcpy(out,d_out,size,cudaMemcpyDeviceToHost);
//     // for(int i=0;i<m;i++){
//     //     for(int j=0;j<n;j++){
//     //         std::cout<<a[i][j]<<", ";
//     //     }
//     //     std::cout<<std::endl;
//     // }
//     for(int i=0;i<m;i++){
//         for(int j=0;j<n;j++){
//             std::cout<<out[i][j]<<", ";
//         }
//         std::cout<<std::endl;
//     }
//     std::cout<<"the total size is "<<size<<" the size of int is "<<sizeof(int)<<std::endl;

// }

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
        v[i]=s->second.first;//1

    }
}
__device__ void check_width(Singleinterval *s, Scalar* v, int dim){
    VectorMax3d w=width(s);
    for(int i=0;i<8;i++){
        
        v[i]=w.v[dim];
    }
}
__device__ Scalar* pter(){
    Scalar *p=new Scalar[8];
    for(int i=0;i<8;i++){
        
        p[i]=i*0.5;
    }
    return p;
}
__device__ void check_return_pointer(Scalar *v){
    //v=pter();
    Scalar* tem=pter();
    //*v=*tem=;
    // for(int i=0;i<8;i++){
        
    //     v[i]=tem[i];
    // }
}   
__device__ void ptr_to_ptr(Singleinterval* p1, int* v){
    Singleinterval *tmp=&p1[1];
    for(int i=0;i<8;i++){
        v[i]=tmp->second.first;//2
    }
}
__device__ void test_bool(bool in, bool &out){
    out=in;
}
__device__ void test_bool_1(bool input, int* v){
    bool tmp;
    test_bool(input,tmp);
    for(int i=0;i<8;i++){
        v[i]=tmp;
    }
}

//__device__ void ptr_and_

__global__ void test_cvt(Scalar* v0,Scalar* v1,Scalar* v2,Scalar* v3,Scalar* v4,Scalar* v5 , int *v6){
    Singleinterval itv[3];
    Numccd zero, one, two, half;
    zero.first=0;zero.second=0;
    one.first=1;one.second=0;
    two.first=2;two.second=0;
    half.first=1; half.second=1;
    Singleinterval s,s1,s2;//[0,1], [0,2],[0,0.5]
    s.first=zero;
    s.second=one;
    s1.first=zero;
    s1.second=two;
    s2.first=zero;
    s2.second=half;
    itv[0]=s;
    itv[1]=s1;
    itv[2]=s2;
    Singleinterval *i0, *i1, *i2;
    i0=new Singleinterval(zero,one);
    i1=new Singleinterval(zero,one);
    i2=new Singleinterval(zero,one);
    
    
    // v6[0]=1;
    // v6[1]=1;
    // check_width(itv,v0,0);
    // check_width(itv,v1,1);
    // check_width(itv,v2,2);
    //convert_tuv_to_array(i0,i1,i2,v0,v1,v2,v3,v4,v5);
    //check(i0,v6);
    //check_return_pointer(v0);
    //ptr_to_ptr(itv,v6);
    test_bool_1(0,v6);
    
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
    //std::cout<<v6[0]<<" "<<v6[1]<<std::endl;
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

__device__ void vf_test_wrapper(CCDdata* vfdata, bool &result){
    
    Scalar* err=new Scalar[3]; err[0]=-1;err[1]=-1;err[2]=-1;
    Scalar ms=0;
    Scalar toi;
    Scalar tolerance=1e-6;
    Scalar t_max=1;
    int max_itr=1e6;
    Scalar output_tolerance;
    bool no_zero_toi=false;
    int overflow_flag;

   result= vertexFaceCCD_double(vfdata,err,ms,toi,tolerance,
    t_max,max_itr,output_tolerance,no_zero_toi,overflow_flag);
}
__global__ void run_parallel_vf(CCDdata* data, bool* res, int size, Scalar* debug){
    debug[0]=0;
    debug[1]=1;
    debug[2]=2;
    int tx=threadIdx.x;
    if(tx<size){
        CCDdata* input=&data[tx];
        vf_test_wrapper(input,res[tx]);
    }
} 

CCDdata array_to_ccd(std::array<std::array<Scalar,3>,8> a){
    CCDdata data;
    for(int i=0;i<3;i++){
        data.v0s[i]=a[0][i];
        data.v1s[i]=a[1][i];
        data.v2s[i]=a[2][i];
        data.v3s[i]=a[3][i];
        data.v0e[i]=a[4][i];
        data.v1e[i]=a[5][i];
        data.v2e[i]=a[6][i];
        data.v3e[i]=a[7][i];
    }
    return data;
}
void test_single_ccd(){
    int dnbr=1;
    
    std::array<std::array<Scalar,3>,8> adata;
    adata[0]={{0,0,0}};
    adata[1]={{0,0,0}};
    adata[2]={{1,0,0}};
    adata[3]={{0,1,0}};
    adata[4]={{-1,-1,-1}};
    adata[5]={{0,0,0}};
    adata[6]={{1,0,0}};
    adata[7]={{0,1,0}};
    CCDdata converted=array_to_ccd(adata);
    CCDdata* vfdata=&converted;
    {
        // just for test
        CCDdata* temp=&vfdata[0];
        std::cout<<"testing "<<vfdata->v0e[0]<<std::endl;
    }


    CCDdata* d_data;
    Scalar* debug=new Scalar[8];
    Scalar* d_debug;
    bool *results=new bool[dnbr], *d_results;
    int data_size=sizeof(CCDdata)*dnbr;
    int result_size=sizeof(bool)*dnbr;
    cudaMalloc(&d_data,data_size);
    cudaMalloc(&d_results,result_size);
    cudaMalloc(&d_debug,int(8*sizeof(Scalar)));
    cudaMemcpy(d_data,vfdata,data_size,cudaMemcpyHostToDevice);
    std::cout<<"before calling"<<std::endl;
    run_parallel_vf<<<1,1>>>(d_data, d_results,dnbr,d_debug);
    std::cout<<"after calling"<<std::endl;
    cudaMemcpy(results,d_results,result_size,cudaMemcpyDeviceToHost);
    std::cout<<"copied 0"<<std::endl;
    cudaMemcpy(debug,d_debug,int(8*sizeof(Scalar)),cudaMemcpyDeviceToHost);
    std::cout<<"copied 1"<<std::endl;
    std::cout<<"vf result is "<<results[0]<<std::endl;
    print_vector(debug,8);
}


int main(int argc, char ** argv){
    test_single_ccd();
    //run_test();
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
