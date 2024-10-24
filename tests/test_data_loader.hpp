#include <chrono>
#include <tahoma.h>
#include <tahoma/data.h>
#include <tahoma/config.h>
#include <tahoma/train/stats_counter.h>

using namespace tahoma;

namespace tahoma::tests {
    int test_config_parse(std::vector<std::string> args) {
        auto config_file = args[0];
        auto config = tahoma::config::Config(config_file);
        std::cerr << "Config: " << config << std::endl;
        return 0;
    }

    int test_read_lines(std::vector<std::string> args) {
        auto config_file = args[0];
        auto config = tahoma::config::Config(config_file);
        auto data_loader = data::DataLoader(config);
        auto paths = config["trainer"]["data"].as<std::vector<std::string>>();
        auto lines = data_loader.read_lines(paths);
        auto max_lines = 1000;
        for (auto line : lines) {
            if (max_lines-- == 0) {
                break;
            }
        }
        return 0;
    }

    int test_read_examples(std::vector<std::string> args) {
        auto config_file = args[0];
        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        auto paths = config["trainer"]["data"].as<std::vector<std::string>>();
        auto max_lengths = config["trainer"]["max_length"].as<std::vector<size_t>>();
        auto max_lines = 1000;
        auto lines = data_loader.read_lines(paths);
        auto examples = data_loader.read_examples(std::move(lines), max_lengths, true);
        for (auto ex : examples) {
            if (max_lines-- == 0) {
                break;
            }
        }
        return 0;
    }

     int test_data_loader_sync(std::vector<std::string> args) {
        auto config_file = args[0];
        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        int32_t nthreads = 8;
        size_t max_batches = 5'000;
        size_t batch_count = 0;
        auto batches = data_loader.get_data_sync("trainer");
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
        // TODO: parse the current_los_message for rate and check it is positive
        return 0;
    }

    int test_data_loader_async(std::vector<std::string> args) {
        auto config_file = args[0];
        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        int32_t nthreads = 8;
        size_t max_batches = 5'000;
        size_t batch_count = 0;
        auto batches = data_loader.get_data_async("trainer", nthreads);
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
        // TODO: parse the current_los_message for rate and check it is positive
        return 0;
    }

    int test_samples(std::vector<std::string> args) {
        auto config_file = args[0];
        auto num_samples = 5;
        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        auto paths = config["validator"]["data"].as<std::vector<std::string>>();
        auto samples = data_loader.get_samples(paths, num_samples);
        return samples.size() == num_samples ? 0 : 1;
    }

}  // namespace tahoma::tests