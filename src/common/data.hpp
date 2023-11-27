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
#include <rtg.hpp>
#include "config.hpp"
#include "utils.hpp"


namespace nn = torch::nn;
namespace optim = torch::optim;
namespace fs = std::filesystem;
namespace sp = sentencepiece;

using namespace std;
using namespace torch::indexing;
using namespace rtg;

namespace rtg::data {

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

        auto read_examples(vector<string> data_paths, vector<size_t> max_length, bool max_length_crop=true) -> std::generator<data::Example> {
            LOG::info("Loading data from {}", fmt::join(data_paths, ","));
            if (data_paths.empty()) {
                throw runtime_error("No data files specified");
            }
            if (max_length_crop){
                if(max_length.size() != data_paths.size()) {
                    throw runtime_error("max_length must be of the same size as data_paths");
                }
                LOG::info("Length cropping is enabled. max_length: {}", fmt::join(max_length, ","));
            } else {
                LOG::info("Length cropping is disabled. Currently long examples are NOT skipped and hence may cause OOM");
            }
            const int32_t num_fields = data_paths.size();
            vector<ifstream> files(num_fields);
            for (size_t i = 0; i < num_fields; ++i) {
                files[i].open(data_paths[i]);
                if (!files[i]) {
                    throw runtime_error("Failed to open file " + data_paths[i]);
                }
            }

            bool contiguous = true;
            vector<string> fields;
            vector<vector<int32_t>> field_ids;
            int64_t rec_num = 0;
            bool has_data = true;
            do {
                fields = vector<string>(num_fields);
                field_ids = vector<vector<int32_t>>(num_fields);

                for (size_t i = 0; i < num_fields; i++) {
                    if (!getline(files[i], fields[i]) || fields[i].empty()) {
                        has_data = false;
                        spdlog::warn("file {} has no more data or there are empty rows. Stopping", data_paths[i]);
                        break;
                    }
                }

                if (has_data) {
                    bool skip = false;
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
            } while (has_data);

            // I wish there was a finally{} block to guarantee file closure :(
            for (auto& file : files) {
                file.close();
            }
            spdlog::info("Reached the end of data files");
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

        auto get_data_sync(string dataset_name) -> std::generator<data::Batch> {
            auto data_paths = config[dataset_name]["data"].as<vector<string>>();
            auto batch_size = config[dataset_name]["batch_size"].as<int>();
            auto max_length_crop = config[dataset_name]["max_length_crop"].as<bool>(true);
            auto max_length = config[dataset_name]["max_length"].as<vector<size_t>>();

            return make_batches(read_examples(data_paths, max_length, max_length_crop),
                 batch_size);
        }

        auto get_data_async(string dataset_name) -> generator<data::Batch> {

            auto data_paths = this->config[dataset_name]["data"].as<vector<string>>();
            auto batch_size = this->config[dataset_name]["batch_size"].as<int>();
            auto max_length_crop = this->config[dataset_name]["max_length_crop"].as<bool>(true);
            auto max_length = this->config[dataset_name]["max_length"].as<vector<size_t>>();

            std::mutex mutex;
            std::condition_variable cv;
            bool producer_done = false;
            std::queue<data::Batch> queue;
            size_t max_queue_size = 24;
            size_t thread_sleep_ms = 4;  // in ms

            std::thread producer_thread([&] { 
                auto batches = make_batches(
                    read_examples(data_paths, max_length, max_length_crop),
                    batch_size);
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