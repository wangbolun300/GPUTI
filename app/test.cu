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
    debug[6] = is_edge;
    
    if (is_edge)
    {
        debug[7] = 1;
        result = edgeEdgeCCD_double(data, err, ms, toi, tolerance,
                                    t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag);
    }
    else
    {
        debug[7] = 2;
        result = vertexFaceCCD_double(data, err, ms, toi, tolerance,
                                      t_max, max_itr, output_tolerance, no_zero_toi, overflow_flag);
    }
    debug[0] = result;
    debug[3] = overflow_flag;
    delete[] err;
}
__global__ void run_parallel_ccd(CCDdata *data, bool *res, int size, Scalar *debug)
{
    
    for (int i = 0; i < 8; i++)
    {
        debug[i] = 1e-1;
    }
    // debug[0]=0;
    // debug[1]=1;
    // debug[2]=2;
    int tx = threadIdx.x;
    
    if (tx < size)
    {
        CCDdata *input = &data[tx];
        bool result;
        
        single_test_wrapper(input, result, debug);
        return;
        res[tx] = result;
        debug[1] = 100 + res[tx];
    }
}

bool single_ccd_run(const std::array<std::array<Scalar, 3>, 8> &V, bool is_edge)
{
    int dnbr = 1;
    // formate convert
    CCDdata converted = array_to_ccd(V, is_edge);
    // host
    CCDdata *data = &converted;
    bool *results = new bool[dnbr];
    Scalar *debug = new Scalar[8];

    // {
    //     // just for test
    //     CCDdata* temp=&data[0];
    //     std::cout<<"testing "<<data->v0e[0]<<" "<<data->is_edge <<std::endl;
    // }
    // device
    CCDdata *d_data;
    bool *d_results;
    Scalar *d_debug;
    // sizes
    int data_size = sizeof(CCDdata) * dnbr;
    int result_size = sizeof(bool) * dnbr;
    // std::cout<<"result size "<<result_size<<std::endl;
    
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_results, result_size);
    cudaMalloc(&d_debug, int(8 * sizeof(Scalar)));
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    
    run_parallel_ccd<<<1, 1>>>(d_data, d_results, dnbr, d_debug);
    return true;
    cudaMemcpy(results, d_results, result_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(debug, d_debug, int(8 * sizeof(Scalar)), cudaMemcpyDeviceToHost);
    bool res = results[0];
    cudaFree(d_data);
    cudaFree(d_results);
    cudaFree(d_debug);
    // free(data); free(results);
#ifdef GPUTI_SHOW_INFO
    std::cout << "in func, result " << res << std::endl;
    std::cout << "\ndebug info " << std::endl;
    print_vector(debug, 8);
#endif
    delete[] results;
    //delete data;
    delete[] debug;
    // delete d_results;
    // delete d_data;
    // delete d_debug;
    return res;

    // formate convert
}
bool single_ccd_run_info(const std::array<std::array<Scalar, 3>, 8> &V, bool is_edge)
{
    int dnbr = 1;
    // formate convert
    CCDdata converted = array_to_ccd(V, is_edge);
    // host
    CCDdata *data = &converted;
    bool *results = new bool[dnbr];
    Scalar *debug = new Scalar[8];
    CCDdata *d_data;
    bool *d_results;
    Scalar *d_debug;
    // sizes
    int data_size = sizeof(CCDdata) * dnbr;
    int result_size = sizeof(bool) * dnbr;
    // std::cout<<"result size "<<result_size<<std::endl;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_results, result_size);
    cudaMalloc(&d_debug, int(8 * sizeof(Scalar)));
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);

    run_parallel_ccd<<<1, 1>>>(d_data, d_results, dnbr, d_debug);

    cudaMemcpy(results, d_results, result_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(debug, d_debug, int(8 * sizeof(Scalar)), cudaMemcpyDeviceToHost);
    bool res = results[0];
    cudaFree(d_data);
    cudaFree(d_results);
    cudaFree(d_debug);
    // free(data); free(results);

    std::cout << "in func, result " << res << std::endl;
    std::cout << "\ndebug info " << std::endl;
    print_vector(debug, 8);
    delete[] results;
    //delete data;
    delete[] debug;
    // delete d_results;
    // delete d_data;
    // delete d_debug;
    return res;

    // formate convert
}
bool WRITE_STATISTIC = true;
bool DEBUG_FLAG = false;
bool DEBUG_FLAG2 = false;
void run_rational_data_single_method(
    const Args &args,
    const bool is_edge_edge,
    const bool is_simulation_data, const std::string folder = "", const std::string tail = "")
{
    std::vector<std::array<Scalar, 3>> all_V;
    std::vector<bool> results;
    ccd::Timer timer;
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
    std::vector<double> tois;
    int fpcounter = 0;

    long long queue_size_total = 0;
    const std::vector<std::string> &scene_names = is_simulation_data ? simulation_folders : handcrafted_folders;
    // if(WRITE_ALL_TIMING){
    //     pout.open(folder+"edge_"+std::to_string(is_edge_edge)+"simu_"+std::to_string(is_simulation_data)+tail+"_all_info.csv");
    //     pout<<"result,timing\n";
    // }
    // std::ofstream fout;
    // fout.open("/home/bolun1/ee"+std::to_string(is_edge_edge)+"simu"+std::to_string(is_simulation_data)+".csv");
    // fout<<"v,v,v,nbr,toi"<<std::endl;
    std::vector<std::string> bases = file_path_base();
    for (const auto &scene_name : scene_names)
    {
        std::string scene_path = args.data_dir + scene_name + sub_folder;
        // if (!fs::exists(scene_path)) {
        //     std::cout << "Missing: " << scene_path << std::endl;
        //     continue;
        // }
        bool skip_folder = false;
        for (const auto &entry : bases)
        {
            // if (entry.path().extension() != ".csv") {
            //     continue;
            // }
            if (skip_folder)
            {
                break;
            }

            // std::cout << "reading data from " << entry.path().string()
            //           << std::endl;
            std::string filename = scene_path + sub_name + "-" + entry + ".csv";

            std::string debug_file = "/home/gameandwatch/bolun/float_with_gt/chain/vertex-face/vertex-face-0010.csv";
            int debug_id = 7459;
            if (DEBUG_FLAG)
            {
                if (filename != debug_file)
                {
                    continue;
                }
            }
            if(DEBUG_FLAG2){
                if (filename != "/home/gameandwatch/bolun/float_with_gt/chain/vertex-face/vertex-face-0010.csv")
                {
                    continue;
                }
            }
            // std::cout<<"filename "<<filename<<std::endl;
            // exit(0);
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
            //assert(all_V.size() % 8 == 0);

            int v_size = all_V.size() / 8;
            for (int i = 0; i < v_size; i++)
            {

                total_number += 1;
                std::array<std::array<Scalar, 3>, 8> V = substract_ccd(all_V, i);
                bool expected_result = results[i * 8];

                bool result;
                timer.start();
                long round_nbr = 0;
                double toi;
                if (DEBUG_FLAG)
                {
                    if (i != debug_id)
                    {
                        continue;
                    }
                }
                for (int ri = 0; ri < 1; ri++)
                {
                    round_nbr += 1;
                    const std::array<double, 3> err = {{-1, -1, -1}};
                    const double t_max = 1;
                    double output_tolerance = args.tight_inclusion_tolerance;
                    result = single_ccd_run(V, is_edge_edge);

                    if(total_number>100){
                        exit(0);
                    }
                    if (DEBUG_FLAG)
                    {
                        std::cout << "checking " << filename << ", id " << i << std::endl;
                        std::cout << "result " << result << ", ground " << expected_result << std::endl;
                        print_V(V);
                        exit(0);
                    }
                    // if (is_edge_edge)
                    // {

                    //     // result=edgeEdgeCCD_OURS( V.row(0), V.row(1), V.row(2), V.row(3), V.row(4),
                    //     //     V.row(5), V.row(6), V.row(7), err, args.minimum_separation,
                    //     //     toi, args.tight_inclusion_tolerance,
                    //     //     t_max, args.tight_inclusion_max_iter, output_tolerance,
                    //     //     1);
                    // }
                    // else
                    // {
                    //     //  result=vertexFaceCCD_OURS( V.row(0), V.row(1), V.row(2), V.row(3), V.row(4),
                    //     //     V.row(5), V.row(6), V.row(7), err, args.minimum_separation,
                    //     //     toi, args.tight_inclusion_tolerance,
                    //     //     t_max, args.tight_inclusion_max_iter, output_tolerance,
                    //     //     1);
                    // }
                    sum_tol += output_tolerance;
                    max_tol = std::max(output_tolerance, max_tol);
                    if (output_tolerance > args.tight_inclusion_tolerance)
                    {
                        nbr_larger_tol++;
                    }
                    if (output_tolerance != args.tight_inclusion_tolerance)
                    {
                        nbr_diff_tol++;
                    }

                    double tt = timer.getElapsedTimeInMicroSec();
                    if (tt > 10)
                        break;
                } // ri

                timer.stop();
                double this_time = timer.getElapsedTimeInMicroSec() / round_nbr;
                total_time += this_time;

                std::cout << total_number << "\r" << std::flush;

                // current_queue_size=inclusion_ccd::return_queue_size();
                // if(current_queue_size>queue_size_max){
                //     queue_size_max=current_queue_size;
                // }
                // queue_sizes.push_back(current_queue_size);

                //queue_size_total+=current_queue_size;

                // if(WRITE_ALL_TIMING){
                //     pout<<result<<","<<this_time<<"\n";
                // }
                if (expected_result)
                {
                    if (DEBUG_FLAG2)
                        {
                            global_counter++;
                            std::cout<<filename<<std::endl;
                            std::cout<<"nbr "<<i<<std::endl;
                            std::cout << "\nfp reproduce" << std::endl;
                            bool tmp_res = single_ccd_run_info(V, is_edge_edge);
                            std::cout << "tmp result= " << tmp_res<<", gt "<< expected_result<< std::endl;
                            
                            if (global_counter > 0)
                            {
                                exit(0);
                            }
                        }
                    total_positives++;
                }
                if (result != expected_result)
                {
                    if (result)
                    {
                        num_false_positives++;
                        if (DEBUG_FLAG2)
                        {
                            global_counter++;
                            std::cout<<filename<<std::endl;
                            std::cout << "\nfp reproduce" << std::endl;
                            bool tmp_res = single_ccd_run_info(V, is_edge_edge);
                            std::cout << "tmp result= " << tmp_res << std::endl;
                            
                            if (global_counter > 1)
                            {
                                exit(0);
                            }
                        }

                        tois.push_back(toi); // we care about FPs' toi
                        // for(int row=0;row<8;row++){
                        //     fout<<std::setprecision(17)<<V(row,0)<<","<<V(row,1)<<","<<V(row,2)<<","<<tois.size()-1<<","<<toi<<std::endl;
                        // }

                        // if(fpcounter==681){
                        //     for(int row=0;row<8;row++){
                        //     std::cout<<V(row,0)<<","<<V(row,1)<<","<<V(row,2)<<","<<tois.size()-1<<","<<toi<<std::endl;
                        //     }
                        //     std::cout<<"pro file "<<entry.path().string()<<"which query "<<i<<std::endl;

                        // }
                        // fpcounter++;
                    }
                    else
                    {
                        num_false_negatives++;
                        std::cout << "false negative happens, " << filename << " nbr " << i
                                  << " out of " << all_V.size() << std::endl;
                        std::cout << "total nbr, p and fp " << total_number << " " << total_positives
                                  << " " << num_false_positives << std::endl;
                        print_V(V);
                        std::cout << "\nreproduce" << std::endl;
                        bool tmp_res = single_ccd_run_info(V, is_edge_edge);
                        std::cout << "tmp result= " << tmp_res << std::endl;
                        exit(0);
                    }
                }
                if (time_upper < this_time)
                {
                    time_upper = this_time;
                    //std::cout<<"upper get updated, "<<time_upper<<std::endl;
                }
                if (time_lower > this_time)
                {
                    time_lower = this_time;
                }
                // write_format qif;
                // qif.method = int(method);
                // qif.nbr = i;
                // qif.result = result;
                // qif.ground_truth = expected_result;
                // qif.is_edge_edge = is_edge_edge;
                // qif.time = this_time;
                // qif.file = entry.path().string();
                // queryinfo.push_back(qif);
            }
        }
    }
    //fout.close();
    // if(WRITE_ALL_TIMING){
    //     pout.close();
    //     std::string fff="file got wrote ";
    //     std::string print_info=fff+
    //     "edge_"+std::to_string(is_edge_edge)+"simu_"+std::to_string(is_simulation_data)+tail+
    //     "_all_info.csv";
    //     std::cout<<print_info<<std::endl;
    // }
    if (WRITE_STATISTIC)
    {
        write_summary(
            folder + "method" + std::to_string(int(2021)) + "_is_edge_edge_" + std::to_string(is_edge_edge) + "_" + std::to_string(total_number + 1) + tail + ".csv",
            2021, total_number + 1, total_positives, is_edge_edge,
            num_false_positives, num_false_negatives,
            total_time / double(total_number + 1), time_lower, time_upper);
    }

    // if(WRITE_QUERY_INFO){
    //  write_results_csv(
    // folder + "method" + std::to_string(method) + "_is_edge_edge_"
    //     + std::to_string(is_edge_edge) + "_"
    //     + std::to_string(total_number + 1) + "_queries" + tail + ".csv",
    // queryinfo);
    // }
    //     if (WRITE_ITERATION_INFO) {
    //     write_iteration_info(
    //         folder + "method" + std::to_string(method) + "_is_edge_edge_"
    //             + std::to_string(is_edge_edge) + "_"
    //             + std::to_string(total_number + 1) + "_itration" + tail
    //             + ".csv",
    //         double(nbr_diff_tol) / total_number, max_tol,
    //         sum_tol / total_number);
    // }
    // if(1){
    //     std::cout<<"start writting queue info"<<std::endl;
    //     queue_size_avg=queue_size_total/(total_number + 1);
    //     std::cout<<"check pt"<<std::endl;
    //     //std::vector<std::string> titles={{"max","avg"}};
    //     std::cout<<"max avg "<<queue_size_max<<" "<<queue_size_avg<<std::endl;
    //     // write_queue_sizes(folder + "method" + std::to_string(method) + "_is_edge_edge_"
    //     //         + std::to_string(is_edge_edge) + "_"
    //     //         + std::to_string(total_number + 1) + "_queue_info" + tail
    //     //         + ".csv",queue_sizes);
    //     std::vector<std::string> titles;

    //     // write_csv(folder + "method" + std::to_string(method) + "_is_edge_edge_"
    //     //         + std::to_string(is_edge_edge) + "_"
    //     //         + std::to_string(total_number + 1) + "_queue_info" + tail
    //     //         + ".csv",titles,queue_sizes,true);
    //     // write_csv(folder + "method" + std::to_string(method) + "_is_edge_edge_"
    //     //         + std::to_string(is_edge_edge) + "_"
    //     //         + std::to_string(total_number + 1) + "_tois" + tail
    //     //         + ".csv",titles,tois,true);
    // }
}

void run_one_method_over_all_data(const Args &args,
                                  const std::string folder = "", const std::string tail = "")
{
    if (args.run_handcrafted_dataset)
    {
        std::cout << "Running handcrafted dataset:\n";
        if (args.run_vf_dataset)
        {
            std::cout << "Vertex-Face:" << std::endl;
            run_rational_data_single_method(
                args, /*is_edge_edge=*/false, /*is_simu_data=*/false, folder, tail);
        }
        if (args.run_ee_dataset)
        {
            std::cout << "Edge-Edge:" << std::endl;
            run_rational_data_single_method(
                args, /*is_edge_edge=*/true, /*is_simu_data=*/false, folder, tail);
        }
    }
    if (args.run_simulation_dataset)
    {
        std::cout << "Running simulation dataset:\n";
        if (args.run_vf_dataset)
        {
            std::cout << "Vertex-Face:" << std::endl;
            run_rational_data_single_method(
                args, /*is_edge_edge=*/false, /*is_simu_data=*/true, folder, tail);
        }
        if (args.run_ee_dataset)
        {
            std::cout << "Edge-Edge:" << std::endl;
            run_rational_data_single_method(
                args, /*is_edge_edge=*/true, /*is_simu_data=*/true, folder, tail);
        }
    }
}
void run_ours_float_for_all_data()
{
    std::string folder = "/home/gameandwatch/bolun/data0809/"; // this is the output folder
    std::string tail = "";

    // tolerance.push_back("1");
    Args arg;
    arg.data_dir = "/home/gameandwatch/bolun/float_with_gt/";

    arg.minimum_separation = 0;
    arg.tight_inclusion_tolerance = 1e-6;
    arg.tight_inclusion_max_iter = 1e6;
    arg.run_ee_dataset = false;
    arg.run_vf_dataset = true;
    arg.run_simulation_dataset = true;
    arg.run_handcrafted_dataset = false;
    run_one_method_over_all_data(arg, folder, tail);

    // run_one_method_over_all_data(arg, CCDMethod::TIGHT_INCLUSION,folder,tail);
}
int main(int argc, char **argv)
{
    run_ours_float_for_all_data();
    std::cout << "done!" << std::endl;
    return 1;
}
