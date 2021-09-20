#include <gputi/io.h>

std::vector<std::string> simulation_folders = {{"chain", "cow-heads", "golf-ball", "mat-twist"}};
std::vector<std::string> handcrafted_folders = {{"erleben-sliding-spike", "erleben-spike-wedge",
                                                 "erleben-sliding-wedge", "erleben-wedge-crack", "erleben-spike-crack",
                                                 "erleben-wedges", "erleben-cube-cliff-edges", "erleben-spike-hole",
                                                 "erleben-cube-internal-edges", "erleben-spikes", "unit-tests"}};

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


// using namespace std;

void read_rational_binary(
   const std::string& inputFileName, std::vector<bool>& results
)
{
    results.clear();
    std::vector<std::array<Scalar, 3>> vs;
    vs.clear();
    std::ifstream infile (inputFileName, std::ios::in | std::ios::binary);
    infile.open(inputFileName);
    std::array<Scalar, 3> v;

    while (!infile.eof())
    {
        infile.read(reinterpret_cast<char*>(&v[0]), sizeof(Scalar)*3);
        vs.push_back(v);
    }
}

void toBinary(
    std::string filename,
    std::vector<std::array<Scalar, 3>>& all_V
)
{
    std::ofstream myFile (filename, std::ios::out | std::ios::binary);
    // Prefer container.data() over &container[0]
    myFile.write (reinterpret_cast<char*>(all_V.data()), all_V.size()*sizeof(Scalar)*3);
}


void csv_to_binarystream(
    const Args &args,
    const bool is_edge_edge,
    std::vector<std::array<Scalar, 3>>& all_V,
    const std::string folder = "", const std::string tail = ""
    )
{
    // arg.run_simulation_dataset = true;
    // arg.run_handcrafted_dataset = false;
    std::string sub_folder = is_edge_edge ? "/edge-edge/" : "/vertex-face/";
    std::string sub_name = is_edge_edge ? "edge-edge" : "vertex-face";

    const std::vector<std::string> &scene_names = args.run_simulation_dataset ? simulation_folders : handcrafted_folders;
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

            std::vector<bool> results;
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

            std::string binaryFilename =  scene_path + sub_name + "-" + entry + ".bin";
            toBinary(binaryFilename, all_V);
        }
    }
}

