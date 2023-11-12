#pragma once

#include <iostream>
#include <coroutine>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

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

        vector<torch::Tensor> fields;
        vector<Example> examples;

        Batch(vector<torch::Tensor> fields, vector<Example> examples)
            : fields(fields), examples(examples) {}

        static Batch from_buffer(vector<Example> buffer) {
            /**
             * Convert a buffer of Examples to a Batch
            */
            int32_t batch_size = buffer.size();
            assert(batch_size > 0);
            int32_t num_fields = buffer[0].field_ids.size();

            vector<int32_t> max_lens(num_fields, 0);
            for (const auto& ex : buffer) {
                for (size_t i = 0; i < num_fields; ++i) {
                    max_lens[i] = std::max(max_lens[i], (int32_t)ex.field_ids[i].size());
                }
            }

            vector<torch::Tensor> fields(num_fields);
            int64_t pad_id = 0;  // pad token id
            for (size_t i = 0; i < num_fields; ++i) {
                fields[i] = torch::full({ batch_size, max_lens[i] }, pad_id, torch::kLong);
                for (int32_t j = 0; j < buffer.size(); ++j) {
                    auto ids = torch::tensor(buffer[j].field_ids[i], torch::kLong);
                    fields[i].index_put_({ j, Slice(0,  buffer[j].field_ids[i].size()) }, ids);
                }
            }
            return Batch(fields, buffer);
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

    struct DataLoader {
        /**
         * A data loader that reads data from a list of files
         * and returns batches of data
        */
        vector<std::shared_ptr<sp::SentencePieceProcessor>> vocabs;
        config::Config config;

        DataLoader(config::Config config, std::vector<std::shared_ptr<sp::SentencePieceProcessor>> vocabs)
            : vocabs(vocabs), config(config)
        {}

        DataLoader(config::Config config)
            : DataLoader(config, load_vocabs(config))
        {}

        ~DataLoader() {}

        auto get_train_data() -> generator<data::Batch> {
            auto data_paths = config["trainer"]["data"].as<vector<string>>();
            auto batch_size = config["trainer"]["batch_size"].as<int>();
            return get_data(data_paths, batch_size);
        }

        auto get_data(vector<string> data_paths, int64_t batch_size) -> generator<data::Batch> {
            LOG::info("Loading data from {}", fmt::join(data_paths, ","));
            assert(batch_size > 0);
            const int32_t num_fields = data_paths.size();
            vector<ifstream> files(num_fields);
            for (size_t i = 0; i < num_fields; ++i) {
                files[i].open(data_paths[i]);
                if (!files[i]) {
                    throw runtime_error("Failed to open file " + data_paths[i]);
                }
            }

            int64_t rec_num = 0;
            vector<data::Example> buffer;
            vector<string> fields;
            vector<vector<int32_t>> field_ids;
            while (true) {
                fields = vector<string>(num_fields);
                field_ids = vector<vector<int32_t>>(num_fields);
                bool has_ended = false;  // in any file
                for (size_t i = 0; i < num_fields; i++) {
                    if (!getline(files[i], fields[i]) || fields[i].empty()) {
                        has_ended = true;
                        spdlog::warn("file {} has no more data or there are empty rows. Stopping", data_paths[i]);
                        break;
                    }
                }
                if (has_ended) { break; }

                bool skip = false;
                for (size_t i = 0; i < num_fields; ++i) {
                    auto ids = vocabs[i]->EncodeAsIds(fields[i]);
                    skip = skip || ids.empty();
                    field_ids[i] = ids;
                }
                if (skip) {
                    spdlog::warn("Skipping empty record {}", rec_num);
                    continue;
                }

                auto ex = data::Example(rec_num, fields, field_ids);
                buffer.push_back(ex);
                rec_num++;
                if (buffer.size() >= batch_size) {
                    co_yield data::Batch::from_buffer(buffer);
                    buffer = vector<data::Example>();
                }
            }

            if (!buffer.empty()) {
                co_yield data::Batch::from_buffer(buffer);
            }
            // I wish there was a finally{} block to guarantee file closure :(
            for (auto& file : files) {
                file.close();
            }
        }

    };

}