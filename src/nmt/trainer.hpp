#include <iostream>
#include <coroutine>
#include <fstream>
#include <ranges>
#include <__generator.hpp>  //reference implementation of std::generator
#include <torch/torch.h>
#include <sentencepiece_processor.h>
#include "../common/utils.hpp"
#include "../common/commons.hpp"

namespace nn = torch::nn;
namespace optim = torch::optim;
namespace fs = std::filesystem;
namespace sp = sentencepiece;

namespace rtg::trainer {

    struct TrainerOptions {
        std::vector<std::string>& data_paths;
        std::vector<std::string>& vocab_paths;
        int64_t epochs;
        int64_t batch_size;
    };



    struct Batch {
        //std::vector<torch::Tensor>& fields;
        //Batch(std::vector<torch::Tensor>& fields): fields(fields) {}

        std::vector<std::vector<int>>& fields;
        std::vector<std::string>& raw_fields;
        Batch(std::vector<std::vector<int>>& fields, std::vector<std::string>& raw_fields): fields(fields), raw_fields(raw_fields) {}

        ~Batch() {}
    };

    class Trainer {
    protected:
        nn::AnyModule& model;
        optim::Optimizer& optimizer;
        optim::LRScheduler& scheduler;
        nn::AnyModule& criterion;
        TrainerOptions& options;
        std::vector<sp::SentencePieceProcessor*> vocabs;

    public:

        static auto load_vocabs(TrainerOptions& options) -> std::vector<sp::SentencePieceProcessor*> {
            auto vocab_paths = options.vocab_paths;
            // SentencePieceProcessor is not copyable and movable, so we use pointers
            std::vector<sp::SentencePieceProcessor*> sps;
            for (auto vocab_path : vocab_paths) {
                spdlog::debug("loading vocab {}", vocab_path);
                auto sp = new sp::SentencePieceProcessor();
                if (!fs::exists(vocab_path)) {
                    spdlog::error("Vocab file {} not found", vocab_path);
                    throw std::runtime_error("Vocab file " + vocab_path + " not found");
                }
                if (!sp->Load(vocab_path).ok()) {
                    throw std::runtime_error("Unable to load vocab from " + vocab_path);
                }
                sps.push_back(sp);
            }
            /**
             * C++ newbie question: sps is a local variable, but we return it by value.
             * So, here the sps is copied? Isnt it a heavy operation?
             *   Alternative: return as a pointer to obj on heap. But thats complicated because we need to manage the memory.
             * Answer: Turns out, std::vector<> has move semantics. The sps is not copied, but moved. Move is cheap.
            */
            return sps;
        }

        Trainer(nn::AnyModule model,
            optim::Optimizer& optimizer,
            optim::LRScheduler& scheduler,
            nn::AnyModule criterion, 
            TrainerOptions& options):
            model(model),
            optimizer(optimizer),
            scheduler(scheduler),
            criterion(criterion),
            options(options),
            vocabs(load_vocabs(options)) {
                if (options.data_paths.size() == 0 || 
                    options.data_paths.size() != options.vocab_paths.size() ||
                    options.data_paths.size() != vocabs.size()) {
                    throw std::runtime_error("Number of data files, vocab files, and vocab objects must be same and > 0");
                }
            }

        ~Trainer() {
            for (auto vocab : vocabs) {
                delete vocab;
            }
        }

        auto get_train_data(TrainerOptions& options) -> std::generator<Batch> {
            auto batch_size = options.batch_size;
            auto data_paths = options.data_paths;
            std::vector<std::ifstream> files(data_paths.size());
            for (size_t i = 0; i < data_paths.size(); ++i) {
                files[i].open(data_paths[i]);
                if (!files[i]) {
                    throw std::runtime_error("Failed to open file " + data_paths[i]);
                }
            }

            while (true) {
                std::vector<std::string> lines(data_paths.size());
                std::vector<std::vector<int>> input_ids(data_paths.size());
                bool has_data = false;  // in all files
                for (size_t i = 0; i < data_paths.size(); ++i) {
                    if (std::getline(files[i], lines[i])) {
                        has_data = true;
                        vocabs[i]->Encode(lines[i], &input_ids[i]);
                    } else {
                        break;
                    }
                }
                if (!has_data) {
                    break;
                }
                co_yield Batch(input_ids, lines);
            }

            for (auto& file : files) {
                file.close();
            }
        }

        void train(TrainerOptions& options) {
            std::cout << "Trainer train\n";
            int64_t step_num = 0;
            for (int64_t epoch = 0; epoch < options.epochs; epoch++) {
                auto train_data = get_train_data(options);
                for (auto batch : train_data) {
                    std::cout << "epoch: " << epoch << " step: " << step_num << " IDs:" << batch.fields << " ||Raw:" << batch.raw_fields << "\n";
                    step_num++;
                }
            }
        }
    };

}

