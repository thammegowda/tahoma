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
namespace fs = std::filesystem;

using namespace std;
using namespace torch::indexing;
using namespace rtg;


//torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

namespace rtg::train {

    template <typename M, typename C>
    class Trainer {
    protected:
        fs::path work_dir;
        config::Config config;
        M model;
        C criterion;
        std::shared_ptr<optim::Optimizer> optimizer;
        std::shared_ptr<optim::LRScheduler> scheduler;
        vector<std::shared_ptr<sp::SentencePieceProcessor>> vocabs;
        torch::Device device { torch::cuda::is_available() ? "cuda" :"cpu" };
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

        static auto init_model(config::Config& config, torch::Device& device) -> M {
            auto model_type = config["model"]["name"].as<string>();
            if (model_type == "transformer") {
                YAML::Node model_args = config["model"]["args"];
                auto model = nmt::transformer::TransformerNMT(model_args);
                //model->to(device);
                return model;
            } else {
                throw runtime_error("Unknown model type " + model_type);
            }
        }

        static auto init_criterion(const config::Config& config) -> C {
            return nn::CrossEntropyLoss(nn::CrossEntropyLossOptions().reduction(torch::kNone));
        }

        static auto init_optimizer(const config::Config& config, M model)
             -> std::shared_ptr<optim::Optimizer> {
            auto optim_config = config["optimizer"];
            auto optim_name = optim_config["name"].as<string>();
            return std::make_shared<optim::Adam>(model->parameters(), optim::AdamOptions(0.0001));
        }

        static auto init_scheduler(const config::Config& config, optim::Optimizer& optimizer)
             -> std::shared_ptr<optim::LRScheduler> {
            return std::make_shared<optim::StepLR>(optimizer, 1.0, 0.95);
        }
        static auto init_config(fs::path work_dir, fs::path config_file) -> config::Config {
            /**
             * 1. If config_file is not provided, look for config.yaml in work_dir
             * 2. If config_file is provided, copy it to work_dir and use it
             * 3. If config_file is not provided and config.yaml is not found in work_dir, raise error
            */
            auto work_config = work_dir / "config.yaml";
            if (!config_file.empty()) { // given non empty config_file
                if (!fs::is_regular_file(config_file)) {
                    throw runtime_error(fmt::format("Config file {} not found", config_file.string()));
                }    
                if (!fs::exists(work_dir)){
                    spdlog::info("mkdir {}", work_dir);
                    fs::create_directories(work_dir);
                }
                spdlog::info("Copy {} ➡️ {}", config_file, work_config);
                fs::copy(work_config, config_file, fs::copy_options::overwrite_existing);
            }
            if (!fs::exists(work_config)) {
                throw runtime_error(fmt::format("Config file {} not found", work_config.string()));
            }
            return config::Config(config_file);
        }

        Trainer(fs::path work_dir, config::Config conf):
            work_dir {work_dir},
            config {conf},
            model {init_model(config, device)},
            criterion {init_criterion(config)},
            optimizer {init_optimizer(config, model)},
            scheduler {init_scheduler(config, *optimizer)},
            vocabs {load_vocabs(config)}
        {
            spdlog::info("Trainer initialized; work_dir={} device = {}", work_dir, device == torch::kCUDA ? "cuda" : "cpu");
        }

        Trainer(fs::path work_dir, fs::path config_file)
        : Trainer(work_dir, init_config(work_dir, config_file))
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
            auto model = this->model;
            model->train();
            model->to(device);
            LOG::info("Moving to device {}", device == torch::kCUDA ? "cuda" : "cpu");

            spdlog::info("Training started; total epochs = {}", num_epochs);
            int64_t step_num = 0;
            for (int32_t epoch = 0; epoch < num_epochs; epoch++) {
                auto train_data = get_train_data();
                for (auto batch : train_data) {
                    batch = batch.to(device);
                    auto src_ids = batch.fields[0];  // [batch_size, seq_len]
                    auto tgt_ids = batch.fields[1];   // [batch_size, seq_len]
                    auto pad_id = 0;
                    auto src_mask = (src_ids == pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
                    auto tgt_mask = (tgt_ids == pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]   // todo: make autoregressive mask

                    auto output = model(src_ids, tgt_ids, src_mask, tgt_mask);
                    auto output_flat = output.view({output.size(0) * output.size(1), -1}); // [batch_size * seq_len, vocab_size]
                    auto tgt_ids_flat = tgt_ids.view({-1}); // [batch_size * seq_len]
                    auto loss = criterion(output_flat, tgt_ids_flat);  // [batch_size * seq_len]
                    //exclude padding tokens from loss calculation
                    loss.masked_fill_(tgt_ids_flat == pad_id, 0.0);
                    auto normalizer = (tgt_ids_flat != pad_id).sum().item(); // #total - #mask
                    //assert(normalizer > 0.0);
                    loss = loss.sum() / normalizer;
                    //assert(!std::isnan(loss.item<float>()));
                    cout << "loss: " << loss.item() << "; sents: " << src_ids.sizes()[0] << "; tokens: " << normalizer << "\n";

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
    auto config_file_arg = args.get<std::string>("config");
    fs::path config_file;
    if (!config_file_arg.empty()) {
        config_file = fs::path {config_file_arg};
    }
    auto trainer = train::Trainer<nmt::transformer::TransformerNMT, nn::CrossEntropyLoss>(work_dir, config_file);
    trainer.train();
    spdlog::info("main finished..");
    return 0;
}

