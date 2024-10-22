
#include <iostream>
#include <chrono>
#include <tahoma.h>
#include <tahoma/data.h>
#include <tahoma/config.h>
#include <tahoma/train/stats_counter.h>

using namespace tahoma;
int main(int argc, char const* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }
    tahoma::global_setup();
    std::string config_file = argv[1];
    spdlog::info("Loading config from {}", config_file);
    auto config = tahoma::config::Config(config_file);
    spdlog::info("Loading data loader");
    auto data_loader = tahoma::data::DataLoader(config);


    /*
    read_lines: 442k lines / sec
    read_examples: 4k/sec on single thread. 30k/sec on 8 threads.  8+ threads has -ve effect
    shuffling overhead is minimal.
    */
   /*
    auto dataset_name = "trainer";
    auto data_paths = config[dataset_name]["data"].as<std::vector<std::string>>();
    auto mini_batch = config[dataset_name]["mini_batch"].as<i32>();
    auto maxi_batch = config[dataset_name]["maxi_batch"].as<i32>(1);
    auto max_length_crop = config[dataset_name]["max_length_crop"].as<bool>(true);
    auto max_length = config[dataset_name]["max_length"].as<vector<size_t>>();
    //auto rows = data_loader.read_lines(data_paths);
    int num_threads = 8;
    auto rows = data_loader.read_examples(data_loader.read_lines(data_paths), max_length,
        max_length_crop, num_threads);
    rows = data_loader.buffered_shuffle(std::move(rows), mini_batch * maxi_batch);
    auto counter = tahoma::train::SimpleCounter();
    for (const auto row: rows ){
        counter.incrememt();
        if (counter.count % 100 == 0 ) {
            std::cout << "Rate :: " << counter.rate() << "\n";
        }
    }
    std::cout << "Final rate:: " << counter.rate() << "\n";
    */
    int32_t nthreads = 8;
    size_t max_batches = 20'000;
    size_t batch_count = 0;
    auto batches = data_loader.get_data_async("trainer", nthreads );
    auto counter = tahoma::train::StatsCounter();
    for (auto batch : batches) {
        batch_count++;
        size_t total_tokens = 0;
        for (const auto& ex : batch.examples) {
            total_tokens += ex.field_ids[0].size();
        }
        counter.update(0.0, batch.size(), total_tokens, 0.0);
        if (batch_count % 500 == 0) {
            std::cout << counter.current_log_message() << std::endl;
        }
        if (batch_count > max_batches) {
            break;
        }
    }
    auto message = counter.current_log_message();
    std::cout << "FINAL:\n\t" << message << std::endl;
    return 0;
}
