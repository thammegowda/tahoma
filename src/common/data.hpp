#pragma once

#include <iostream>
#include <coroutine>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

#include <__generator.hpp>  //reference implementation of generator
#include <torch/torch.h>
#include <sentencepiece_processor.h>
#include <tahoma.hpp>
#include "config.hpp"
#include "utils.hpp"


namespace nn = torch::nn;
namespace optim = torch::optim;
namespace fs = std::filesystem;
namespace sp = sentencepiece;

using namespace std;
using namespace torch::indexing;
using namespace tahoma;

namespace tahoma::data {

    struct Example {

        int64_t id;
        vector<string> fields;
        vector<vector<int32_t>> field_ids;

        Example(int64_t id, vector<string> fields, vector<vector<int32_t>> field_ids)
            : id(id), fields(fields), field_ids(field_ids) {}

        ~Example() {}

        //copy ctr
        Example(const Example& other)
            : id(other.id), fields(other.fields), field_ids(other.field_ids) {}

        //copy assignment
        Example& operator=(const Example& other) {
            if (this != &other) {
                id = other.id;
                fields = other.fields;
                field_ids = other.field_ids;
            }
            return *this;
        }

        // move ctr
        Example(Example&& other) noexcept :
            id(other.id), fields(other.fields), field_ids(other.field_ids) {
        }

        friend std::ostream& operator<<(std::ostream& os, const Example& example);

    };

    // operator<<
    std::ostream& operator<<(std::ostream& os, const Example& ex) {
        os << "Example(" << ex.id << "; fields(" <<
            ex.fields.size() << "): " << ex.fields
            << "; ids: (" << ex.field_ids.size() << "))";
        return os;
    }

    struct Batch {

        vector<Example> examples;
        vector<torch::Tensor> fields = {};

        Batch(vector<Example> examples, bool contiguous = false)
            : examples(examples)
        {
            if (contiguous) {
                this->contiguous();
            }
        }

        static vector<torch::Tensor> to_tensors(vector<Example> examples) {
            /**
             * Convert a buffer of Examples to a Batch
            */
            int32_t batch_size = examples.size();
            assert(batch_size > 0);
            int32_t num_fields = examples[0].field_ids.size();

            vector<int32_t> max_lens(num_fields, 0);
            for (const auto& ex : examples) {
                for (size_t i = 0; i < num_fields; ++i) {
                    max_lens[i] = std::max(max_lens[i], (int32_t)ex.field_ids[i].size());
                }
            }

            vector<torch::Tensor> fields(num_fields);
            int64_t pad_id = 0;  // pad token id
            for (size_t i = 0; i < num_fields; ++i) {
                fields[i] = torch::full({ batch_size, max_lens[i] }, pad_id, torch::kLong);
                for (int32_t j = 0; j < examples.size(); ++j) {
                    auto ids = torch::tensor(examples[j].field_ids[i], torch::kLong);
                    fields[i].index_put_({ j, Slice(0,  examples[j].field_ids[i].size()) }, ids);
                }
            }
            return fields;
        }

        void contiguous() {
            /**
             * Convert vector of Examples to contiguous tensors with padding
             * The reason we do this on-demand: since batches are queued, we want to avoid filling up memory with padded tensors.
            */
            if (fields.size() > 0) {
                // already contiguous
                return;
            }
            auto temp = to_tensors(examples);
            fields.insert(fields.end(), temp.begin(), temp.end());
        }

        auto to(torch::Device& device) -> Batch& {
            for (auto idx = 0; idx < fields.size(); ++idx) {
                fields[idx] = fields[idx].to(device);
            }
            return *this;
        }

        auto size() -> int32_t {
            return examples.size();
        }

        ~Batch() {}
    };

    auto load_vocabs(const config::Config& config) -> vector<std::shared_ptr<sp::SentencePieceProcessor>> {
        auto vocab_paths = config["schema"]["vocabs"].as<std::vector<std::string>>();
        assert(!vocab_paths.empty()); // expected atleast one vocabulary
        // SentencePieceProcessor is not copyable and movable, so we use pointers
        vector<std::shared_ptr<sp::SentencePieceProcessor>> spps;
        for (auto vocab_path : vocab_paths) {
            spdlog::debug("loading vocab {}", vocab_path);
            auto spp = std::make_shared<sp::SentencePieceProcessor>();
            if (!fs::exists(vocab_path)) {
                spdlog::error("Vocab file {} not found", vocab_path);
                throw runtime_error("Vocab file " + vocab_path + " not found");
            }
            if (!spp->Load(vocab_path).ok()) {
                throw runtime_error("Unable to load vocab from " + vocab_path);
            }
            spps.push_back(spp);
        }
        return spps;
    }


    struct DataLoader {  // ideally, DataGenerator but it could be confusing as synthetic data generator
        /**
         * A data loader that reads data from a list of files
         * and returns batches of data
        */
        vector<std::shared_ptr<sp::SentencePieceProcessor>> vocabs;
        config::Config config;

        DataLoader(config::Config config, std::vector<std::shared_ptr<sp::SentencePieceProcessor>> vocabs)
            : vocabs(vocabs), config(config)
        {
        }

        DataLoader(config::Config config)
            : DataLoader(config, load_vocabs(config))
        {}

        ~DataLoader() {}

        auto read_lines(vector<string> data_paths) -> std::generator<vector<string>> {
             if (data_paths.empty()) {
                throw runtime_error("No data files specified");
            }
            LOG::info("Loading data from {}", fmt::join(data_paths, ", "));
            const i32 num_fields = data_paths.size();
            vector<ifstream> files(num_fields);
            for (size_t i = 0; i < num_fields; ++i) {
                files[i].open(data_paths[i]);
                if (!files[i]) {
                    throw runtime_error("Failed to open file " + data_paths[i]);
                }
            }

            bool has_data = true;
            int64_t rec_num = 0;
            do {
                auto fields = vector<string>(num_fields);
                rec_num++;
                for (size_t i = 0; i < num_fields; i++) {
                    if (!getline(files[i], fields[i])) {
                        has_data = false;
                        break;
                    }
                    if (fields[i].empty()) {
                        throw runtime_error("Empty line in file: " + data_paths[i] + " at line: " + std::to_string(rec_num));
                    }
                }

                if (has_data) {
                    co_yield fields;
                }
            } while (has_data);

            // wish there was a finally{} block to guarantee file closure :(
            for (auto& file : files) {
                file.close();
            }
            spdlog::debug("Reached the end of data files");
        }

        auto read_examples(std::generator<vector<string>> rows, vector<size_t> max_length,
                bool max_length_crop=true) -> std::generator<data::Example> {
            i32 num_fields = -1;
            i64 rec_num = 0;
            for (auto fields: rows) {
                if (num_fields < 0) {
                    num_fields = fields.size(); // initialize on first run
                    if (max_length_crop){
                        if(max_length.size() != num_fields) {
                            throw runtime_error("max_length must be of the same size as data_paths");
                        }
                        LOG::info("Length cropping is enabled. max_length: {}", fmt::join(max_length, ", "));
                    }
                }
                if (fields.size() != num_fields) {
                    throw runtime_error(fmt::format("All data files must have the same number of fields. Record number {} has {} fields, expected {}", rec_num, fields.size(), num_fields));
                }
                bool skip = false;
                auto field_ids = vector<vector<int32_t>>(num_fields);
                 for (size_t i = 0; i < num_fields; ++i) {
                    auto ids = vocabs[i]->EncodeAsIds(fields[i]);
                    if (max_length_crop && ids.size() > max_length[i]) {
                        ids = vector<int32_t>(ids.begin(), ids.begin() + max_length[i]);
                    }
                    skip = skip || ids.empty();
                    field_ids[i] = ids;
                }
                if (skip) {
                    spdlog::warn("Skipping empty record {}", rec_num);
                    continue;
                }
                co_yield data::Example(rec_num, fields, field_ids);
                rec_num++;
            }            
        }

        auto make_batches(std::generator<data::Example> examples, size_t batch_size,
            bool contiguous = false) -> std::generator<data::Batch> {
            // TODO: buffer and batch equal length examples to reduce padding
            vector<data::Example> buffer;
            for (auto ex : examples) {
                buffer.push_back(ex);
                if (buffer.size() >= batch_size) {
                    co_yield data::Batch(buffer, contiguous);
                    buffer = vector<data::Example>();
                }
            }
            if (!buffer.empty()) {
                co_yield data::Batch(buffer, contiguous);
            }
        }

        auto get_train_data(size_t n_data_threads=1) -> generator<data::Batch> {
            if (n_data_threads >= 1) { // asynchronous, on a separate thread
                LOG::info("Using async loader with {} data threads", n_data_threads);
                // TODO: support multiple threads
                return get_data_async("trainer");
            } else if (n_data_threads == 0) { // synchronous, on the same thread
                LOG::info("Data loading on the main thread");
                return get_data_sync("trainer");
            } else {
                throw runtime_error("n_data_threads must be >= 1");
            }
        }

        auto get_validation_data() -> generator<data::Batch> {
            return get_data_sync("validator", "trainer");
        }

        auto get_samples(vector<string> data_paths, i32 num_samples) -> data::Batch {
            assert (num_samples > 0);
            auto samples = tahoma::utils::sample_n_items<vector<string>>(read_lines(data_paths), num_samples);
            auto examples = read_examples(std::move(samples), {}, false);
            auto batches = make_batches(std::move(examples), num_samples);
            for (auto batch: batches){
                return batch; // first batch
            }
            throw runtime_error("No samples found");
        }

        auto get_data_sync(string dataset_name, string fallback_name="trainer") -> std::generator<data::Batch> {
            auto data_paths = config[dataset_name]["data"].as<vector<string>>();
            // try to locate bacth_size in the dataset_name, else fallback to trainer 
            auto batch_size = config[dataset_name]["batch_size"].as<int>(config[fallback_name]["batch_size"].as<int>());
            auto max_length_crop = config[dataset_name]["max_length_crop"].as<bool>(config[fallback_name]["max_length_crop"].as<bool>(true));
            auto max_length = config[dataset_name]["max_length"].as<vector<size_t>>(config[fallback_name]["max_length"].as<vector<size_t>>());
            auto lines = read_lines(data_paths);
            auto examples = read_examples(std::move(lines), max_length, max_length_crop);
            auto batches = make_batches(std::move(examples), batch_size);
            return batches;
        }

        auto get_data_async(string dataset_name) -> generator<data::Batch> {

            auto data_paths = this->config[dataset_name]["data"].as<vector<string>>();
            auto batch_size = this->config[dataset_name]["batch_size"].as<i32>();
            auto max_length_crop = this->config[dataset_name]["max_length_crop"].as<bool>(true);
            auto max_length = this->config[dataset_name]["max_length"].as<vector<size_t>>();

            std::mutex mutex;
            std::condition_variable cv;
            bool producer_done = false;
            std::queue<data::Batch> queue;
            size_t max_queue_size = 24;
            size_t thread_sleep_ms = 4;  // in ms

            std::thread producer_thread([&] { 
                auto lines = read_lines(data_paths);
                auto examples = read_examples(std::move(lines), max_length, max_length_crop);
                auto batches = make_batches(std::move(examples), batch_size);
                for (auto batch : batches) {
                    while (queue.size() >= max_queue_size) {
                        // wait for the queue to drain
                        std::this_thread::sleep_for(std::chrono::milliseconds(thread_sleep_ms));
                    }
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        queue.push(batch);
                    }
                    cv.notify_one();
                }
                // notify consumers that we are done
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    producer_done = true;
                }
            });

            // producer_thread.detach();
            // read from queue and co_yield
            while (true) {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] {return !queue.empty() || producer_done; });
                if (queue.empty() && producer_done) {
                    break;
                }
                auto batch = queue.front();
                queue.pop();
                lock.unlock();
                //batch.contiguous();
                co_yield batch;
            }
            producer_thread.join();
        }
    };

}