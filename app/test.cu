#include <gputi/root_finder.h>
#include "timer.hpp"
#include <iostream>
#include <fstream>
#include "read_rational_csv.hpp"
// #include <boost/filesystem.hpp>
#include <filesystem>
// #include <experimental/filesystem>
// namespace fs = std::filesystem;
// // namespace fs = std::experimental::filesystem;
int global_counter = 0;

std::vector<std::string> simulation_folders = {{"chain", "cow-heads", "golf-ball", "mat-twist"}};
std::vector<std::string> handcrafted_folders = {{"erleben-sliding-spike", "erleben-spike-wedge",
                                                 "erleben-sliding-wedge", "erleben-wedge-crack", "erleben-spike-crack",
                                                 "erleben-wedges", "erleben-cube-cliff-edges", "erleben-spike-hole",
                                                 "erleben-cube-internal-edges", "erleben-spikes", "unit-tests"}};
struct Args
{
    std::string data_dir;
    double minimum_separation = 0;
    double tight_inclusion_tolerance = 1e-6;
    long tight_inclusion_max_iter = 1e6;
    bool run_ee_dataset = true;
    bool run_vf_dataset = true;
    bool run_simulation_dataset = true;
    bool run_handcrafted_dataset = true;
};
void print_V(std::array<std::array<Scalar, 3>, 8> V)
{
    for (int i = 0; i < 8; i++)
    {
        std::cout << V[i][0] << ", " << V[i][1] << ", " << V[i][2] << std::endl;
        if (i == 3)
        {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}
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
    // fout<<"method, "<<method<<std::endl;
    // fout<<"total nbr, "<<total_number<<std::endl;
    // fout<<"positives, "<<positives<<std::endl;
    // fout<<"is_edge_edge, "<<is_edge_edge<<std::endl;
    // fout<<"fp, "<<fp<<std::endl;
    // fout<<"fn, "<<fn<<std::endl;
    // fout<<"average time, "<<average_time<<std::endl;
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
std::vector<std::string> file_path_base()
{
    // path is in the form of "chain/edge-edge/"
    std::vector<std::string> result;
    result.reserve(9999);
    for (int i = 1; i < 10000; i++)
    {
        std::string base;
        if (i < 10)
        {
            base = "000" + std::to_string(i);
        }
        if (i >= 10 && i < 100)
        {
            base = "00" + std::to_string(i);
        }
        if (i >= 100 && i < 1000)
        {
            base = "0" + std::to_string(i);
        }
        if (i >= 1000 && i < 10000)
        {
            base = std::to_string(i);
        }
        result.push_back(base);
    }
    return result;
}

__device__ void single_test_wrapper(CCDdata *data, bool &result, Scalar *debug)
{
    // TODO write these parameters into CCDdata class

    Scalar *err = new Scalar[3];
    err[0] = -1;
    err[1] = -1;
    err[2] = -1;
    Scalar ms = 0;
    Scalar toi;
    Scalar tolerance = 1e-6;
    Scalar t_max = 1;
    int max_itr = 1e6;
    Scalar output_tolerance;
    bool no_zero_toi = false;
    int overflow_flag;
    //debug[7]=1;
    bool is_edge = data->is_edge;
    //debug[6] = is_edge;
    CCDdata data_cp;
    for (int i = 0; i < 3; i++)
    {
        data_cp.v0s[i] = data->v0s[i];
        data_cp.v1s[i] = data->v1s[i];
        data_cp.v2s[i] = data->v2s[i];
        data_cp.v3s[i] = data->v3s[i];
        data_cp.v0e[i] = data->v0e[i];
        data_cp.v1e[i] = data->v1e[i];
        data_cp.v2e[i] = data->v2e[i];
        data_cp.v3e[i] = data->v3e[i];
    }
    data_cp.is_edge = data->is_edge;

    if (is_edge)
    {
        //debug[7] = 1;
        result = edgeEdgeCCD_double(data_cp, err, ms, toi, tolerance,
                                    t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag);
    }
    else
    {
        //debug[7] = 2;
        result = vertexFaceCCD_double(data_cp, err, ms, toi, tolerance,
                                      t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag);
    }
    //debug[0] = result;
    //debug[3] = overflow_flag;
    delete[] err;
}
__device__ void single_test_wrapper_return_toi(CCDdata *data, bool &result, Scalar &time_impact)
{
    
    Scalar err[3];
    err[0] = -1;
    err[1] = -1;
    err[2] = -1;
    Scalar ms = 0;
    Scalar toi;
    Scalar tolerance = 1e-6;
    Scalar t_max = 1;
    int max_itr = 1e6;
    Scalar output_tolerance;
    bool no_zero_toi = false;
    int overflow_flag;
    bool is_edge = data->is_edge;
    CCDdata data_cp;
    for (int i = 0; i < 3; i++)
    {
        data_cp.v0s[i] = data->v0s[i];
        data_cp.v1s[i] = data->v1s[i];
        data_cp.v2s[i] = data->v2s[i];
        data_cp.v3s[i] = data->v3s[i];
        data_cp.v0e[i] = data->v0e[i];
        data_cp.v1e[i] = data->v1e[i];
        data_cp.v2e[i] = data->v2e[i];
        data_cp.v3e[i] = data->v3e[i];
    }
    data_cp.is_edge = is_edge;
    
#ifdef CHECK_EE
        result = edgeEdgeCCD_double(data_cp, err, ms, toi, tolerance,
                              t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag);
#else
        result = vertexFaceCCD_double(data_cp, err, ms, toi, tolerance,
                                      t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag);
#endif
    time_impact = toi;
    return;
}

__global__ void run_parallel_ccd_all(CCDdata *data, bool *res, int size, Scalar *tois)
{

    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tx < size)
    {
        CCDdata *input = &data[tx];
        bool result;
        Scalar toi;
        single_test_wrapper_return_toi(input, result, toi);
        res[tx] = result;
        tois[tx] = toi;
    }
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
        data_list[i] = array_to_ccd(V[i], is_edge);
    }
    bool *res = new bool[nbr];
    Scalar *tois = new Scalar[nbr];

    // device
    CCDdata *d_data_list;
    bool *d_res;
    Scalar *d_tois;

    int data_size = sizeof(CCDdata) * nbr;
    int result_size = sizeof(bool) * nbr;
    int time_size = sizeof(Scalar) * nbr;

    cudaMalloc(&d_data_list, data_size);
    cudaMalloc(&d_res, result_size);
    cudaMalloc(&d_tois, time_size);
    cudaMemcpy(d_data_list, data_list, data_size, cudaMemcpyHostToDevice);

    // std::cout << "data copied" << std::endl;
    // parallel info

    ccd::Timer timer;

    timer.start();
    run_parallel_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>(d_data_list, d_res, nbr, d_tois);
    cudaDeviceSynchronize();

    //size_test<<<nbr/parallel_nbr+1, parallel_nbr>>>(nbr, d_data_list, d_res, d_tois);
    // naive_test<<<nbr/parallel_nbr+1, parallel_nbr>>>();
    // cudaDeviceSynchronize();
    double tt = timer.getElapsedTimeInMicroSec();
    run_time = tt;

    // std::cout << "finished parallization running , nbr "<<nbr << std::endl;
    // std::cout<<"parallel info, threads and grids "<<parallel_nbr<<" "<<nbr/parallel_nbr+1<<std::endl;
    // std::cout<<"sizes "<<data_size<<" "<<result_size<<" "<<time_size<<std::endl;
    // std::cout<<"size of Scalar "<<sizeof(Scalar)<<std::endl;

    cudaMemcpy(res, d_res, result_size, cudaMemcpyDeviceToHost);

    cudaMemcpy(tois, d_tois, time_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data_list);
    cudaFree(d_res);
    cudaFree(d_tois);

    for (int i = 0; i < nbr; i++)
    {
        result_list[i] = res[i];
    }

    time_impact.resize(nbr);

    for (int i = 0; i < nbr; i++)
    {
        time_impact[i] = tois[i];
    }

    delete[] res;
    delete[] data_list;
    delete[] tois;
    return;
}

bool WRITE_STATISTIC = true;
bool DEBUG_FLAG = false;
bool DEBUG_FLAG2 = false;


void run_rational_data_single_method_parallel(
    const Args &args,
    const bool is_edge_edge,
    const bool is_simulation_data, int parallel, const std::string folder = "", const std::string tail = "")
{
    std::vector<std::array<Scalar, 3>> all_V;
    std::vector<bool> results;

    //std::vector<write_format> queryinfo;
    int total_number = -1;
    double total_time = 0.0;
    int total_positives = 0;
    int num_false_positives = 0;
    int num_false_negatives = 0;
    double time_lower = 1e100;
    double time_upper = -1;
    std::string sub_folder = is_edge_edge ? "/edge-edge/" : "/vertex-face/";
    std::string sub_name = is_edge_edge ? "edge-edge" : "vertex-face";
    int nbr_larger_tol = 0;
    int nbr_diff_tol = 0;
    double max_tol = 0;
    double sum_tol = 0;
    long queue_size_avg = 0;
    long queue_size_max = 0;
    long current_queue_size = 0;
    std::vector<long> queue_sizes;
    std::vector<Scalar> tois;
    int fpcounter = 0;

    std::vector<bool> result_list;
    std::vector<bool> expect_list;
    std::vector<std::array<std::array<Scalar, 3>, 8>> queries;

    long long queue_size_total = 0;
    const std::vector<std::string> &scene_names = is_simulation_data ? simulation_folders : handcrafted_folders;
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

            std::string debug_file = "/home/bolun/bolun/float_with_gt/chain/vertex-face/vertex-face-0010.csv";
            int debug_id = 7459;
            if (DEBUG_FLAG)
            {
                if (filename != debug_file)
                {
                    continue;
                }
            }
            if (DEBUG_FLAG2)
            {
                if (filename != "/home/bolun/bolun/float_with_gt/chain/vertex-face/vertex-face-0010.csv")
                {
                    continue;
                }
                std::cout << "running this file" << std::endl;
            }
            // std::cout<<"filename "<<filename<<std::endl;
            // exit(0);
            if (queries.size() > 1e6)
            {
                break;
            }
            all_V = ccd::read_rational_csv(filename, results);
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
                if (queries.size() > 1e6)
                {
                    break;
                }
                total_number += 1;

                std::array<std::array<Scalar, 3>, 8> V = substract_ccd(all_V, i);
                bool expected_result = results[i * 8];

                bool result;

                double toi;
                if (DEBUG_FLAG)
                {
                    if (i != debug_id)
                    {
                        continue;
                    }
                }
                queries.push_back(V);
                expect_list.push_back(expected_result);
                //result = single_ccd_run(V, is_edge_edge);

                if (DEBUG_FLAG)
                {
                    std::cout << "checking " << filename << ", id " << i << std::endl;
                    std::cout << "result " << result << ", ground " << expected_result << std::endl;
                    print_V(V);
                    exit(0);
                }
            }
        }
    }
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
                // std::cout << "false negative!!!, result " << result_list[i]<<" id "<<i << std::endl;
                // exit(0);
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
    std::string folder = "/home/bolun/bolun/data0809/"; // this is the output folder
    std::string tail = "_prl_" + std::to_string(parallel);

    // tolerance.push_back("1");
    Args arg;
    arg.data_dir = "/home/bolun/bolun/float_with_gt/";

    arg.minimum_separation = 0;
    arg.tight_inclusion_tolerance = 1e-6;
    arg.tight_inclusion_max_iter = 1e6;
    arg.run_ee_dataset = false;
    arg.run_vf_dataset = true;
    arg.run_simulation_dataset = true;
    arg.run_handcrafted_dataset = false;
    run_one_method_over_all_data(arg, parallel, folder, tail);

    // run_one_method_over_all_data(arg, CCDMethod::TIGHT_INCLUSION,folder,tail);
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
    bool a=false;
    bool b=false;
    bool c=a*b;
    std::cout<<"c is "<<c<<std::endl;
    return 1;
}
