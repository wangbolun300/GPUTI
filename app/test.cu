#include <gputi/root_finder.h>
#include <gputi/book.h>
#include "timer.hpp"
#include <iostream>
#include <functional>
#include <fstream>
#include <filesystem>
#include <cuda/std/functional>

#include <gputi/timer.cuh>
#include <gputi/io.h>

using namespace ccd;
extern std::vector<std::string> simulation_folders;
extern std::vector<std::string> handcrafted_folders;

std::array<std::array<Scalar, 3>, 8> substract_ccd(const std::vector<std::array<Scalar, 3>> &data, int nbr)
{
    std::array<std::array<Scalar, 3>, 8> result;
    int start = nbr * 8;
    for (int i = 0; i < 8; i++)
    {
        result[i] = data[i + start];
    }
    return result;
}
void write_summary(
    const std::string file,
    const int method,
    const int total_number,
    const int positives,
    const bool is_edge_edge,
    const int fp,
    const int fn,
    const double average_time,
    const double time_lower,
    const double time_upper)
{
    std::ofstream fout;
    fout.open(file);
    fout << "method,total_nbr,positives,is_edge_edge,fp,fn,average_time, time_lower, time_upper"
         << std::endl;
    fout << method << "," << total_number << "," << positives << ","
         << is_edge_edge << "," << fp << "," << fn << ',' << average_time << "," << time_lower << "," << time_upper
         << std::endl;
    fout.close();
}
template <typename T>
void write_csv(const std::string &file, const std::vector<std::string> titles, const std::vector<T> data, bool large_info)
{
    std::cout << "inside write" << std::endl;
    std::ofstream fout;
    fout.open(file);

    if (large_info)
    {
        fout << "data" << std::endl;
        for (int i = 0; i < data.size(); i++)
        {
            fout << data[i] << std::endl;
        }
    }
    else
    {
        for (int i = 0; i < titles.size() - 1; i++)
        {
            fout << titles[i] << ",";
        }
        fout << titles.back() << std::endl;
        for (int i = 0; i < data.size() - 1; i++)
        {
            fout << data[i] << ",";
        }
        fout << data.back() << std::endl;
    }

    fout.close();
}


__global__ void run_parallel_vf_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= size) return;
    // copy the input queries to __device__
    CCDdata data_in;
    for (int i = 0; i < 3; i++)
    {
        data_in.v0s[i] = data[tx].v0s[i];
        data_in.v1s[i] = data[tx].v1s[i];
        data_in.v2s[i] = data[tx].v2s[i];
        data_in.v3s[i] = data[tx].v3s[i];
        data_in.v0e[i] = data[tx].v0e[i];
        data_in.v1e[i] = data[tx].v1e[i];
        data_in.v2e[i] = data[tx].v2e[i];
        data_in.v3e[i] = data[tx].v3e[i];
    }
    // copy the configurations to the shared memory
    __shared__ CCDConfig config;
    config.err_in[0]=config_in->err_in[0];
    config.err_in[1]=config_in->err_in[1];
    config.err_in[2]=config_in->err_in[2];
    config.co_domain_tolerance=config_in->co_domain_tolerance; // tolerance of the co-domain
    config.max_t=config_in->max_t; // the upper bound of the time interval
    config.max_itr=config_in->max_itr;// the maximal nbr of iterations
    CCDOut out;
    vertexFaceCCD(data_in,config, out);
    res[tx] = out.result;
    tois[tx] = out.toi;
}
__global__ void run_parallel_ee_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= size) return;
    // copy the input queries to __device__
    CCDdata data_in;
    for (int i = 0; i < 3; i++)
    {
        data_in.v0s[i] = data[tx].v0s[i];
        data_in.v1s[i] = data[tx].v1s[i];
        data_in.v2s[i] = data[tx].v2s[i];
        data_in.v3s[i] = data[tx].v3s[i];
        data_in.v0e[i] = data[tx].v0e[i];
        data_in.v1e[i] = data[tx].v1e[i];
        data_in.v2e[i] = data[tx].v2e[i];
        data_in.v3e[i] = data[tx].v3e[i];
    }
    // copy the configurations to the shared memory
    __shared__ CCDConfig config;
    config.err_in[0]=config_in->err_in[0];
    config.err_in[1]=config_in->err_in[1];
    config.err_in[2]=config_in->err_in[2];
    config.co_domain_tolerance=config_in->co_domain_tolerance; // tolerance of the co-domain
    config.max_t=config_in->max_t; // the upper bound of the time interval
    config.max_itr=config_in->max_itr;// the maximal nbr of iterations
    CCDOut out;
    edgeEdgeCCD(data_in,config, out);
    res[tx] = out.result;
    tois[tx] = out.toi;
}


void all_ccd_run(const std::vector<std::array<std::array<Scalar, 3>, 8>> &V, bool is_edge,
                 std::vector<bool> &result_list, double &run_time, std::vector<Scalar> &time_impact, int parallel_nbr)
{
    int nbr = V.size();
    result_list.resize(nbr);
    // host
    CCDdata *data_list = new CCDdata[nbr];
    for (int i = 0; i < nbr; i++)
    {
        data_list[i] = array_to_ccd( V[i]);
    }
    bool *res = new bool[nbr];
    Scalar *tois = new Scalar[nbr];
    CCDConfig *config=new CCDConfig[1];
    config[0].err_in[0]=-1;// the input error bound calculate from the AABB of the whole mesh
    config[0].co_domain_tolerance=1e-6; // tolerance of the co-domain
    config[0].max_t=1; // the upper bound of the time interval
    config[0].max_itr=1e6;// the maximal nbr of iterations

    // device
    CCDdata *d_data_list;
    bool *d_res;
    Scalar *d_tois;
    CCDConfig *d_config;

    int data_size = sizeof(CCDdata) * nbr;
    int result_size = sizeof(bool) * nbr;
    int time_size = sizeof(Scalar) * nbr;
   // int dbg_size=sizeof(Scalar)*8;

    cudaMalloc(&d_data_list, data_size);
    cudaMalloc(&d_res, result_size);
    cudaMalloc(&d_tois, time_size);
    cudaMalloc(&d_config, sizeof(CCDConfig));

    cudaMemcpy(d_data_list, data_list, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_config, config, sizeof(CCDConfig), cudaMemcpyHostToDevice);

    ccd::Timer timer;
    cudaProfilerStart();
     timer.start();
    if(is_edge){
        run_parallel_ee_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
        d_data_list,d_config, d_res, nbr, d_tois);
    }
    else{
        run_parallel_vf_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
        d_data_list,d_config, d_res, nbr, d_tois);
    }
   
    
    cudaDeviceSynchronize();
    double tt = timer.getElapsedTimeInMicroSec();
    run_time = tt;
    cudaProfilerStop();

    cudaMemcpy(res, d_res, result_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(tois, d_tois, time_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(dbg, d_dbg, dbg_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data_list);
    cudaFree(d_res);
    cudaFree(d_tois);
    cudaFree(d_config);
    //cudaFree(d_dbg);

    for (int i = 0; i < nbr; i++)
    {
        result_list[i] = res[i];
    }

    time_impact.resize(nbr);

    for (int i = 0; i < nbr; i++)
    {
        time_impact[i] = tois[i];
    }
    // std::cout << "dbg info\n"
    //           << dbg[0] << "," << dbg[1] << "," << dbg[2] << "," << dbg[3] << "," << dbg[4] << "," << dbg[5] << "," << dbg[6] << "," << dbg[7] << std::endl;
    delete[] res;
    delete[] data_list;
    delete[] tois;
    delete[] config;
    //delete[] dbg;
    cudaError_t ct = cudaGetLastError();
    printf("******************\n%s\n************\n", cudaGetErrorString(ct));
    
    return;
}


bool WRITE_STATISTIC = true;

void run_rational_data_single_method_parallel(
    const Args &args,
    const bool is_edge_edge,
    const bool is_simulation_data, int parallel, const std::string folder = "", const std::string tail = "")
{
    std::vector<std::array<Scalar, 3>> all_V;
    std::vector<bool> results;

    //std::vector<write_format> queryinfo;
    int total_number = -1;
    int total_positives = 0;
    int num_false_positives = 0;
    int num_false_negatives = 0;
    double time_lower = 1e100;
    double time_upper = -1;
    std::string sub_folder = is_edge_edge ? "/edge-edge/" : "/vertex-face/";
    std::string sub_name = is_edge_edge ? "edge-edge" : "vertex-face";
    std::vector<long> queue_sizes;
    std::vector<Scalar> tois;

    std::vector<bool> result_list;
    std::vector<bool> expect_list;
    std::vector<std::array<std::array<Scalar, 3>, 8>> queries;
    const std::vector<std::string> &scene_names = is_simulation_data ? ccd::simulation_folders : ccd::handcrafted_folders;
    std::cout << "loading data" << std::endl;
    std::vector<std::string> bases = file_path_base();
    for (const auto &scene_name : scene_names)
    {
        std::string scene_path = args.data_dir + scene_name + sub_folder;

        bool skip_folder = false;
        for (const auto &entry : bases)
        {
            if (skip_folder)
            {
                break;
            }
            std::string filename = scene_path + sub_name + "-" + entry + ".csv";

            // std::cout<<"filename "<<filename<<std::endl;
            // exit(0);
            if (queries.size() > TEST_NBR_QUERIES)
            {
                break;
            }
            // all_V = ccd::read_rational_csv(filename, results);
            // all_V = read_rational_csv_bin(filename, results);

            std::string filename_noext =  filename.substr(0, filename.find_last_of("."));
            
            std::string vertexFilename = std::string(filename_noext + "_vertex.bin");
            std::ifstream vinfile (vertexFilename, std::ios::in | std::ios::binary);

            std::string resultsFilename = std::string(filename_noext + "_result.bin");
            std::ifstream rinfile (resultsFilename, std::ios::in | std::ios::binary);
            
            if (vinfile && rinfile)
            {
                read_rational_binary(vertexFilename, all_V );
                read_rational_binary(resultsFilename, results );
            }
            else 
                all_V = read_rational_csv_bin(filename, results);
            
            if (all_V.size() == 0)
            {
                std::cout << "data size " << all_V.size() << std::endl;
                std::cout << filename << std::endl;
            }

            if (all_V.size() == 0)
            {
                skip_folder = true;
                continue;
            }

            int v_size = all_V.size() / 8;
            for (int i = 0; i < v_size; i++)
            {
                if (queries.size() > TEST_NBR_QUERIES)
                {
                    break;
                }
                total_number += 1;

                std::array<std::array<Scalar, 3>, 8> V = substract_ccd(all_V, i);
                bool expected_result = results[i * 8];
                queries.push_back(V);
                expect_list.push_back(expected_result);
            }
        }
    }

#ifdef GPUTI_GO_DEAP_HEAP
    std::array<std::array<Scalar, 3>, 8> deep_one = queries[TESTING_ID];
    std::cout<<"query\n";
    for(int i=0;i<8;i++){
        std::cout<<deep_one[i][0]<<", "<<deep_one[i][1]<<", "<<deep_one[i][2]<<std::endl;
        if(i==3){
            std::cout<<std::endl;
        }
    }
    queries.resize(TEST_SIZE);
    expect_list.resize(queries.size());
    for (int i = 0; i < queries.size(); i++)
    {
        queries[i] = deep_one;
    }
#endif
    int size = queries.size();
    std::cout << "data loaded, size " << queries.size() << std::endl;
    double tavg = 0;
    int max_query_cp_size = 1e7;
    int start_id = 0;

    result_list.resize(size);
    tois.resize(size);

    while (1)
    {
        std::vector<bool> tmp_results;
        std::vector<std::array<std::array<Scalar, 3>, 8>> tmp_queries;
        std::vector<Scalar> tmp_tois;

        int remain = size - start_id;
        double tmp_tall;

        if (remain <= 0)
            break;

        int tmp_nbr = min(remain, max_query_cp_size);
        tmp_results.resize(tmp_nbr);
        tmp_queries.resize(tmp_nbr);
        tmp_tois.resize(tmp_nbr);
        for (int i = 0; i < tmp_nbr; i++)
        {
            tmp_queries[i] = queries[start_id + i];
        }
        all_ccd_run(tmp_queries, is_edge_edge, tmp_results, tmp_tall, tmp_tois, parallel);

        tavg += tmp_tall;
        for (int i = 0; i < tmp_nbr; i++)
        {
            result_list[start_id + i] = tmp_results[i];
            tois[start_id + i] = tmp_tois[i];
        }

        start_id += tmp_nbr;
    }
    tavg /= size;
    std::cout << "avg time " << tavg << std::endl;

    if (expect_list.size() != size)
    {
        std::cout << "size wrong!!!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < size; i++)
    {
        if (expect_list[i])
        {
            total_positives++;
        }
        if (result_list[i] != expect_list[i])
        {
            if (expect_list[i])
            {
                num_false_negatives++;
            }
            else
            {
                num_false_positives++;
            }
        }
    }
    std::cout << "total positives " << total_positives << std::endl;
    std::cout << "num_false_positives " << num_false_positives << std::endl;
    std::cout << "num_false_negatives " << num_false_negatives << std::endl;
    total_number = size;
    if (WRITE_STATISTIC)
    {
        write_summary(
            folder + "method" + std::to_string(int(2021)) + "_is_edge_edge_" + std::to_string(is_edge_edge) + "_" + std::to_string(total_number) + tail + ".csv",
            2021, total_number, total_positives, is_edge_edge,
            num_false_positives, num_false_negatives,
            tavg, time_lower, time_upper);
    }

    if (1)
    {
        std::vector<std::string> titles;
        write_csv(folder + "method" + std::to_string(int(2021)) + "_is_edge_edge_" + std::to_string(is_edge_edge) + "_" +
                      std::to_string(total_number) + "_tois" + tail + ".csv",
                  titles, tois, true);

        // write_csv(folder + "method" + std::to_string(int(2021)) + "_is_edge_edge_" + std::to_string(is_edge_edge) + "_" +
        // std::to_string(total_number) + "_runtime" + tail + ".csv", titles, time_list, true);
    }
}

void run_one_method_over_all_data(const Args &args, int parallel,
                                  const std::string folder = "", const std::string tail = "")
{
    if (args.run_handcrafted_dataset)
    {
        std::cout << "Running handcrafted dataset:\n";
        if (args.run_vf_dataset)
        {
            std::cout << "Vertex-Face:" << std::endl;
            run_rational_data_single_method_parallel(
                args, /*is_edge_edge=*/false, /*is_simu_data=*/false, parallel, folder, tail);
        }
        if (args.run_ee_dataset)
        {
            std::cout << "Edge-Edge:" << std::endl;
            run_rational_data_single_method_parallel(
                args, /*is_edge_edge=*/true, /*is_simu_data=*/false, parallel, folder, tail);
        }
    }
    if (args.run_simulation_dataset)
    {
        std::cout << "Running simulation dataset:\n";
        if (args.run_vf_dataset)
        {
            std::cout << "Vertex-Face:" << std::endl;
            run_rational_data_single_method_parallel(
                args, /*is_edge_edge=*/false, /*is_simu_data=*/true, parallel, folder, tail);
        }
        if (args.run_ee_dataset)
        {
            std::cout << "Edge-Edge:" << std::endl;
            run_rational_data_single_method_parallel(
                args, /*is_edge_edge=*/true, /*is_simu_data=*/true, parallel, folder, tail);
            
        }
    }
}
void run_ours_float_for_all_data(int parallel)
{
    std::string folder = std::string(getenv("HOME")) + "/data0809/"; // this is the output folder
    std::string tail = "_prl_" + std::to_string(parallel);
    Args arg;
    arg.data_dir = std::string(getenv("HOME")) + "/float_with_gt/";

    arg.minimum_separation = 0;
    arg.tight_inclusion_tolerance = 1e-6;
    arg.tight_inclusion_max_iter = 1e6;

    arg.run_ee_dataset = true;
    arg.run_vf_dataset = false;
    
    arg.run_simulation_dataset = true;
    arg.run_handcrafted_dataset = false;

    run_one_method_over_all_data(arg, parallel, folder, tail);

}
int main(int argc, char **argv)
{
    // int deviceCount;
    //     cudaGetDeviceCount(&deviceCount);
    //     for(int i=0;i<deviceCount;i++)
    //     {
    //         cudaDeviceProp devProp;
    //         cudaGetDeviceProperties(&devProp, i);
    //         std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
    //         std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
    //         std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    //         std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    //         std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    //         std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
    //         std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    //         std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    //         std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
    //         std::cout << "======================================================" << std::endl;

    //     }
    //     return 0;
    // int alpha=5;
    // double* a = new double[alpha];
    int parallel = 0;
    if (argc == 1)
    {
        parallel = 1;
    }
    else
    {
        parallel = std::stoi(argv[1]);
    }
    if (parallel <= 0)
    {
        std::cout << "wrong parallel nbr = " << parallel << std::endl;
        return 0;
    }

    run_ours_float_for_all_data(parallel);
    std::cout << "done!" << std::endl;
    return 0;
}
