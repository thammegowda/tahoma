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


//torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;


namespace rtg::trainer {

    struct TrainerOptions {
        vector<string>& data_paths;
        vector<string>& vocab_paths;
        int64_t epochs;
        int64_t batch_size;
    };


    struct Batch {
        vector<torch::Tensor>& fields;
        vector<string>& raw_fields;
        Batch(vector<torch::Tensor>& fields, vector<string>& raw_fields) : fields(fields), raw_fields(raw_fields) {}

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
        vector<sp::SentencePieceProcessor*> vocabs;

    public:

        static auto load_vocabs(TrainerOptions& options) -> vector<sp::SentencePieceProcessor*> {
            auto vocab_paths = options.vocab_paths;
            // SentencePieceProcessor is not copyable and movable, so we use pointers
            vector<sp::SentencePieceProcessor*> spps;
            for (auto vocab_path : vocab_paths) {
                spdlog::debug("loading vocab {}", vocab_path);
                auto spp = new sp::SentencePieceProcessor();
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
                    auto msg = fmt::format("Number of data files, vocab files, and vocab objects must be equal and 2 (i.e source, target). data_paths: {}, vocab_paths: {}, vocabs: {}", options.data_paths.size(), options.vocab_paths.size(), vocabs.size());
                    throw runtime_error(msg);
            }
            cerr << model << "\n";
        }

        ~Trainer() {
            for (auto& vocab : vocabs) {
                delete vocab;
            }
        }

        auto get_train_data(TrainerOptions& options) -> generator<Batch> {
            auto batch_size = options.batch_size;
            auto data_paths = options.data_paths;
            vector<ifstream> files(data_paths.size());
            for (size_t i = 0; i < data_paths.size(); ++i) {
                files[i].open(data_paths[i]);
                if (!files[i]) {
                    throw runtime_error("Failed to open file " + data_paths[i]);
                }
            }

            while (true) {
                vector<string> lines(data_paths.size());
                vector<vector<int>> input_ids(data_paths.size());
                
                bool has_data = false;  // in all files
                for (size_t i = 0; i < data_paths.size(); ++i) {
                    if (getline(files[i], lines[i])) {
                        has_data = true;
                        vocabs[i]->Encode(lines[i], &input_ids[i]);
                    } else {
                        spdlog::warn("file {} has no more data. Stopping", data_paths[i]);
                        has_data = false;
                        break;
                    }
                }
                if (!has_data) {
                    break;
                }
                vector<torch::Tensor> input_tensors(data_paths.size());
                for (size_t i = 0; i < data_paths.size(); ++i) {
                    input_tensors[i] = torch::tensor(input_ids[i], torch::kI64).view({ 1, -1 });
                }
                co_yield Batch(input_tensors, lines);
            }

            for (auto& file : files) {
                file.close();
            }
        }

        void train(TrainerOptions& options) {
            cout << "Trainer train\n";
            int64_t step_num = 0;
            for (int64_t epoch = 0; epoch < options.epochs; epoch++) {
                auto train_data = get_train_data(options);
                for (auto batch : train_data) {
                    cout << "epoch: " << epoch << " step: " << step_num << " IDs:" << batch.fields << " ||Raw:" << batch.raw_fields << "\n";
                    auto src_ids = batch.fields[0];
                    auto tgt_ids = batch.fields[1];
                    auto src_key_padding_mask = src_ids == 0;
                    auto tgt_key_padding_mask = tgt_ids == 0;
                    auto output = model(src_ids, tgt_ids);
                    auto loss = criterion(output, tgt_ids);
                    cout << "loss: " << loss << "\n";

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

