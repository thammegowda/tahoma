#include <iostream>
#include <coroutine>
#include <fstream>
#include <ranges>
#include <memory>
#include <__generator.hpp>  //reference implementation of generator
#include <torch/torch.h>
#include <sentencepiece_processor.h>
#include "../common/utils.hpp"
#include "../common/commons.hpp"

namespace nn = torch::nn;
namespace optim = torch::optim;
namespace fs = std::filesystem;
namespace sp = sentencepiece;

using namespace std;
using namespace torch::indexing;


//torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;


namespace rtg::trainer {

    struct TrainerOptions {
        vector<string> data_paths;
        vector<string> vocab_paths;
        int32_t epochs;
        int32_t batch_size;
    };

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
        Example(Example&& other) noexcept:
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
    
        static Batch from_buffer(vector<Example> buffer){
            /**
             * Convert a buffer of Examples to a Batch
            */
            int32_t batch_size = buffer.size();
            assert (batch_size > 0);
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
                fields[i] = torch::full({batch_size, max_lens[i]}, pad_id, torch::kLong);
                for (int32_t j = 0; j < buffer.size(); ++j) {
                    auto ids = torch::tensor( buffer[j].field_ids[i], torch::kLong);
                    fields[i].index_put_({j, Slice(0,  buffer[j].field_ids[i].size())}, ids);
                }
            }
            return Batch(fields, buffer);
        }
        
        ~Batch() {}
    };

    template <typename M, typename C>
    class Trainer {
    protected:
        M model;
        C criterion;
        optim::Optimizer& optimizer;
        optim::LRScheduler& scheduler;
        TrainerOptions& options;
        vector<std::shared_ptr<sp::SentencePieceProcessor>> vocabs;

    public:
        
        static auto load_vocabs(TrainerOptions& options) -> vector<std::shared_ptr<sp::SentencePieceProcessor>> {
            auto vocab_paths = options.vocab_paths;
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

        Trainer(M model,
            C criterion,
            optim::Optimizer& optimizer,
            optim::LRScheduler& scheduler,
            TrainerOptions& options) :
            model(model),
            criterion(criterion),
            optimizer(optimizer),
            scheduler(scheduler),
            options(options),
            vocabs(load_vocabs(options)) {
            if (options.data_paths.size() != 2 ||
                options.data_paths.size() != options.vocab_paths.size() ||
                options.data_paths.size() != vocabs.size()) {
                    auto msg = fmt::format("Number of data files, vocab files, \
                     and vocab objects must be equal and 2 (i.e source, target). \
                     data_paths: {}, vocab_paths: {}, vocabs: {}",
                     options.data_paths.size(), options.vocab_paths.size(), vocabs.size());
                    throw runtime_error(msg);
            }
            cerr << model << "\n";
        }

        ~Trainer() {
            for (auto& vocab : vocabs) {
                //delete vocab;
            }
        }

        auto get_train_data(TrainerOptions& options) -> generator<Batch> {
            auto batch_size = options.batch_size;
            auto data_paths = options.data_paths;
            LOG::info("Loading data from {}", fmt::join(data_paths, ","));
            assert (batch_size > 0);
            const int32_t num_fields = data_paths.size();
            vector<ifstream> files(num_fields);
            for (size_t i = 0; i < num_fields; ++i) {
                files[i].open(data_paths[i]);
                if (!files[i]) {
                    throw runtime_error("Failed to open file " + data_paths[i]);
                }
            }

            int64_t rec_num = 0;
            vector<Example> buffer;
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
                auto ex = Example(rec_num, fields, field_ids);
                buffer.push_back(ex);
                rec_num++;
                if (buffer.size() >= batch_size) {
                    co_yield Batch::from_buffer(buffer);
                    buffer = vector<Example>();
                }
            }

            if (!buffer.empty()) {
                co_yield Batch::from_buffer(buffer);
            }

            for (auto& file : files) {
                file.close();
            }
        }

        void train(TrainerOptions& options) {
            spdlog::info("Training started");
            int64_t step_num = 0;
            for (int32_t epoch = 0; epoch < options.epochs; epoch++) {
                auto train_data = get_train_data(options);
                for (auto batch : train_data) {
                    cout << "epoch: " << epoch << "; step: " << step_num 
                        << "; src: " << batch.fields[0].sizes() << " tgt: " << batch.fields[1].sizes() << "\n";
                    for (auto& ex : batch.examples) {
                        cout << "####" << ex.id << "\t" << ex.fields << "\n";
                    }
                    auto src_ids = batch.fields[0].transpose(0, 1);  // [Seq_len, Batch_size]
                    auto tgt_ids = batch.fields[1].transpose(0, 1);   // [Seq_len, Batch_size]
                    cout << "src_ids: " << src_ids.sizes() << "\t" << "tgt_ids: " << tgt_ids.sizes() << "\n";
                    auto pad_id = 0;
                    auto src_mask = src_ids == pad_id;
                    auto tgt_mask = tgt_ids == pad_id;
                    auto output = model(src_ids, tgt_ids, src_mask, tgt_mask);
                    auto loss = criterion(output, tgt_ids);
                    //cout << "loss: " << loss << "\n";

                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                    scheduler.step();
                    step_num++;
                }
            }
        }
    };

}

