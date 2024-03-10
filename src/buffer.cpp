/*
This file is a a buffer for scratch work.
It is used to test the code snippets from the source code to quickly debug and test the code.
*/

#include <torch/torch.h>
#include "common/config.hpp"
#include "train/utils.hpp"
#include "train/trainer.hpp"

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    cout << "Hello, World!" << endl;
    auto t = torch::arange(0, 12).reshape({3, 4});
    cout << t << endl;

    cout << "Cuda available ?" << torch::cuda::is_available() << endl;
    cout << "Cuda device count ?" << torch::cuda::device_count() << endl;
    cout << "Tensor on cuda ::" << t.to(torch::kCUDA) << endl;


    auto config_file = fs::path("examples/transformer-nmt.yaml");
    auto config = tahoma::config::Config(config_file);
    cout << "Config file ::" << config << endl;

    fs::path work_dir = "tmp.work_dir";
    auto trainer = tahoma::train::Trainer(work_dir, config);

    // sleep  for 5s; so we can see memory usage in nvidia-smi
    std::this_thread::sleep_for(std::chrono::seconds(5s));

    cout << "Training starting.." << endl;
    trainer.train();

    return 0;
}

