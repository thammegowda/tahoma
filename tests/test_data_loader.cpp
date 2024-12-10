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
        auto lines = data::read_lines(paths);
        auto count = 1000;
        for (auto line : lines) {
            if (--count == 0) {
                break;
            }
        }
        return count == 0 ? 0 : 1;
    }
     int test_samples(std::vector<std::string> args) {
        auto config_file = args[0];
        auto num_samples = 5;
        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        auto paths = config["validator"]["data"].as<std::vector<std::string>>();
        auto samples = data_loader.get_samples(paths, num_samples);
        for (auto ex : samples.examples) {
            std::cerr << ex << std::endl;
            num_samples--;
        }
        return num_samples == 0 ? 0 : 1;
    }

    int test_read_examples(std::vector<std::string> args) {
        auto config_file = args[0];

        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        auto paths = config["trainer"]["data"].as<std::vector<std::string>>();
        auto max_lengths = config["trainer"]["max_length"].as<std::vector<size_t>>();
        auto count = 1000;
        auto examples = data_loader.read_examples(paths, max_lengths, true);
        for (auto ex : examples) {
            if (--count == 0) {
                break;
            }
        }
        return count == 0 ? 0 : 1;
    }

    int test_make_batches(std::vector<std::string> args) {
        auto config_file = args[0];
        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        auto paths = config["trainer"]["data"].as<std::vector<std::string>>();
        auto max_lengths = config["trainer"]["max_length"].as<std::vector<size_t>>();
        auto examples = data_loader.read_examples(paths, max_lengths, true);
        auto batch_size = 32;
        auto batches = data_loader.make_batches(examples, batch_size);
        auto count = 1000;
        for (auto batch : batches) {
            if (--count == 0) {
                break;
            }
        }
        return count == 0 ? 0 : 1;
    }

    int test_data_loader_sync_reimpl(std::vector<std::string> args) {
        auto config_file = args[0];
        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        //int32_t nthreads = 8;
        //auto batches = data_loader.get_data_sync("trainer");
        /////

         // TODO remove this once async is stable and bug free
        auto dataset_name = "trainer";
        auto fallback_name = "trainer";
        auto data_paths = config[dataset_name]["data"].as<std::vector<std::string>>();
        // try to locate batch_size in the dataset_name, else fallback to trainer
        auto mini_batch = config[dataset_name]["mini_batch"].as<int>(config[fallback_name]["maxi_batch"].as<int>());
        auto maxi_batch = config[dataset_name]["maxi_batch"].as<int>(config[fallback_name]["mini_batch"].as<int>());
        auto max_length_crop = config[dataset_name]["max_length_crop"].as<bool>(config[fallback_name]["max_length_crop"].as<bool>(true));
        auto max_length = config[dataset_name]["max_length"].as<vector<size_t>>(config[fallback_name]["max_length"].as<vector<size_t>>());

        spdlog::info("Loading data from {}", fmt::join(data_paths, ", "));
        spdlog::info("mini_batch: {}, maxi_batch: {}", mini_batch, maxi_batch);
        spdlog::info("max_length_crop: {}, max_length: {}", max_length_crop, fmt::join(max_length, ", "));

        auto examples = data_loader.read_examples(data_paths, max_length, max_length_crop);
        auto examples_shufd = data_loader.buffered_shuffle(examples, mini_batch * 10);
        auto batches = data_loader.make_batches(examples_shufd, mini_batch);

        /////
        size_t batch_count = 0;
        size_t max_batches = 2'000;
        auto counter = tahoma::train::StatsCounter();
        for (auto batch : batches) {
            batch_count++;
            size_t total_tokens = 0;
            for (const auto& ex : batch.examples) {
                total_tokens += ex.field_ids[0].size();
            }
            counter.update(0.0, batch.size(), total_tokens, 0.0);
            if (batch_count % 250 == 0) {
                std::cerr << counter.current_log_message() << std::endl;
            }
            if (batch_count >= max_batches) {
                break;
            }
        }
        // TODO: parse the current_log_message for rate and check it is positive
        return batch_count == max_batches ? 0 : 1;
    }

    int test_data_loader_sync(std::vector<std::string> args) {
        auto config_file = args[0];
        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        auto n_data_threads = 0;
        auto batches = data_loader.get_train_data(n_data_threads);

        /////
        size_t batch_count = 0;
        size_t max_batches = 2'000;
        auto counter = tahoma::train::StatsCounter();
        for (auto batch : batches) {
            batch_count++;
            size_t total_tokens = 0;
            for (const auto& ex : batch.examples) {
                total_tokens += ex.field_ids[0].size();
            }
            counter.update(0.0, batch.size(), total_tokens, 0.0);
            if (batch_count % 250 == 0) {
                std::cerr << counter.current_log_message() << std::endl;
            }
            if (batch_count >= max_batches) {
                break;
            }
        }
        // TODO: parse the current_log_message for rate and check it is positive
        return batch_count == max_batches ? 0 : 1;
    }

    int test_data_loader_async(std::vector<std::string> args) {
        auto config_file = args[0];
        auto config = tahoma::config::Config(config_file);
        auto data_loader = tahoma::data::DataLoader(config);
        int32_t nthreads = 2;
        size_t max_batches = 2'000;
        size_t batch_count = 0;
        auto batches = data_loader.get_data_async_new("trainer", nthreads);
        auto counter = tahoma::train::StatsCounter();
        for (auto batch : batches) {
            batch_count++;
            size_t total_tokens = 0;
            for (const auto& ex : batch.examples) {
                total_tokens += ex.field_ids[0].size();
            }
            counter.update(0.0, batch.size(), total_tokens, 0.0);
            if (batch_count % 500 == 0) {
                std::cerr << counter.current_log_message() << std::endl;
            }
            if (batch_count >= max_batches) {
                break;
            }
        }
        // TODO: parse the current_log_message for rate and check it is positive
        return batch_count == max_batches ? 0 : 1;
    }

}  // namespace tahoma::tests