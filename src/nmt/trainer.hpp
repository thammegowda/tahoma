#pragma once
#include <iostream>
#include <coroutine>
#include <ranges>
#include <memory>
#include <__generator.hpp>  //reference implementation of generator
#include <argparse.hpp>
#include <torch/torch.h>
#include <sentencepiece_processor.h>

#include "../common/utils.hpp"
#include "../common/commons.hpp"
#include "../common/config.hpp"
#include "../common/data.hpp"
#include "../nmt/transformer.hpp"


namespace nn = torch::nn;
namespace optim = torch::optim;
namespace sp = sentencepiece;

using namespace std;
using namespace torch::indexing;
using namespace rtg;


//torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

namespace rtg::train {

    template <typename M, typename C>
    class Trainer {
    protected:
        config::Config config;
        M model;
        C criterion;
        std::shared_ptr<optim::Optimizer> optimizer;
        std::shared_ptr<optim::LRScheduler> scheduler;
        vector<std::shared_ptr<sp::SentencePieceProcessor>> vocabs;

    public:
        
        static auto load_vocabs(const config::Config& config) -> vector<std::shared_ptr<sp::SentencePieceProcessor>> {
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

        static auto init_model(config::Config& config) -> M {
            auto model_type = config["model"]["type"].as<string>();
            if (model_type == "transformer") {
                return nmt::transformer::TransformerNMT(config);
            } else {
                throw runtime_error("Unknown model type " + model_type);
            }
        }

        static auto init_criterion(const config::Config& config) -> C {
            return nn::CrossEntropyLoss();
        }

        static auto init_optimizer(const config::Config& config, M model) -> std::shared_ptr<optim::Optimizer> {
            return std::make_shared<optim::Adam>(model->parameters(), optim::AdamOptions(0.0001));
        }

        static auto init_scheduler(const config::Config& config, optim::Optimizer& optimizer) -> std::shared_ptr<optim::LRScheduler> {
            return std::make_shared<optim::StepLR>(optimizer, 1.0, 0.95);
        }

        Trainer(config::Config& conf):
            config {conf},
            model {init_model(config)},
            criterion {init_criterion(config)},
            optimizer {init_optimizer(config, model)},
            scheduler {init_scheduler(config, *optimizer)},
            vocabs {load_vocabs(config)}
        {}

        ~Trainer() {
            for (auto& vocab : vocabs) {
                //delete vocab;
            }
        }

        auto get_train_data() -> generator<data::Batch> {
            
            auto batch_size = config["trainer"]["batch_size"].as<int>();
            auto data_paths = config["trainer"]["data"].as<vector<string>>();
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

        void train() {
            size_t num_epochs = config["trainer"]["epochs"].as<int>();

            spdlog::info("Training started; total epochs = {}", num_epochs);
            int64_t step_num = 0;
            for (int32_t epoch = 0; epoch < num_epochs; epoch++) {
                auto train_data = get_train_data();
                for (auto batch : train_data) {
                    cout << "epoch: " << epoch << "; step: " << step_num 
                        << "; src: " << batch.fields[0].sizes() << " tgt: " << batch.fields[1].sizes() << "\n";
                    for (auto& ex : batch.examples) {
                        cout << "####" << ex.id << "\t" << ex.fields << "\n";
                    }
                    //auto src_ids = batch.fields[0].transpose(0, 1);  // [Seq_len, Batch_size]
                    //auto tgt_ids = batch.fields[1].transpose(0, 1);   // [Seq_len, Batch_size]
                    auto src_ids = batch.fields[0];  // [batch_size, seq_len]
                    auto tgt_ids = batch.fields[1];   // [batch_size, seq_len]
                    cout << "src_ids: " << src_ids.sizes() << "\t" << "tgt_ids: " << tgt_ids.sizes() << "\n";
                    auto pad_id = 0;
                    auto src_mask = (src_ids == pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
                    auto tgt_mask = (tgt_ids == pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]   // todo: make autoregressive mask

                    auto output = model(src_ids, tgt_ids, src_mask, tgt_mask);
                    auto output_flat = output.view({output.size(0) * output.size(1), -1}); // [batch_size * seq_len, vocab_size]
                    auto tgt_ids_flat = tgt_ids.view({-1}); // [batch_size * seq_len]
                    cout << "output_flat: " << output_flat.sizes() << "\t" << "tgt_ids_flat: " << tgt_ids_flat.sizes() << "\n";
                    auto loss = criterion(output_flat, tgt_ids_flat); // TODO: exclude padding tokens
                    cout << "loss: " << loss << "\n";

                    optimizer->zero_grad();
                    loss.backward();
                    optimizer->step();
                    scheduler->step();
                    step_num++;

                    if (step_num >  20) {
                        std::cout << "Training aborted manually....\n";
                        return;
                    }
                }
            }
        }
    };

    auto parse_args(int argc, char* argv[]) -> argparse::ArgumentParser {
        argparse::ArgumentParser parser("trainer");
        parser.add_argument("-v", "--verbose").help("Increase log verbosity")
            .default_value(false).implicit_value(true);
        parser.add_argument("work_dir").help("Working directory").required();
        parser.add_argument("-c", "--config").help("Config file. Optional: default is config.toml in work_dir");

        try {
            parser.parse_args(argc, argv);
        }
        catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::cerr << parser;
            exit(1);
        }
        return parser;
    }
}


int main(int argc, char* argv[]) {
    

    auto args = train::parse_args(argc, argv);
    if (args.get<bool>("verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }

    auto work_dir = fs::path {args.get<std::string>("work_dir")};
    spdlog::info("work_dir: {}", work_dir);
    auto config_file_arg = args.get<std::string>("config");
    
    namespace fs = std::filesystem;
    fs::path config_file =  work_dir / "config.yaml";
    if (!fs::exists(work_dir)){
        spdlog::info("Creating work dir {}", work_dir);
        fs::create_directories(work_dir);
    }
    if (!config_file_arg.empty()){
        spdlog::info("copying config file {} -> {}", config_file_arg, config_file);
        fs::copy(fs::path(config_file_arg), config_file, fs::copy_options::overwrite_existing);
    }

    if (!fs::exists(config_file)) {
        spdlog::error("config file {} not found", config_file);
        throw std::runtime_error("config file" + std::string(config_file) + "not found");
    }

    auto config = rtg::config::Config(config_file);
    std::cout << "Config:\n\n" << config << "\n";
    auto trainer = train::Trainer<nmt::transformer::TransformerNMT, nn::CrossEntropyLoss>(config);
    trainer.train();
    spdlog::info("main finished..");
    return 0;
}

