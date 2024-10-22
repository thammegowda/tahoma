/*
This file is a buffer for scratch work.
It is used to test the code snippets from the source code to quickly debug and test the code.
*/
#include <tahoma.h>
#include <tahoma/config.h>
#include <tahoma/data.h>

int main(int argc, char* argv[]) {
    std::cout << "Hello, World!\n";
    tahoma::global_setup();
    auto t = torch::arange(0, 12).reshape({3, 4});
    std::cout << t << std::endl;
    std::cout << "Cuda available ?" << torch::cuda::is_available() << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "Cuda device count ?" << torch::cuda::device_count() << std::endl;
        std::cout << "Tensor on cuda ::" << t.to(torch::kCUDA) << std::endl;
    }

    auto config_file = fs::path("examples/transformer-nmt.yaml");
    auto config = tahoma::config::Config(config_file);
    std::cout << "Config file ::" << config << std::endl;
    auto vocabs = tahoma::data::load_vocabs(config);
    auto data_loader = tahoma::data::DataLoader(config, vocabs);
    auto data = data_loader.get_data_async("trainer");
    size_t i = 0;
    for (auto batch : data) {
        i++;
        std::cout << "Batch ::" << i << "\t" << batch.size() << std::endl;
    }
    return 0;
}