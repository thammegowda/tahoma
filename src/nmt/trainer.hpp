#pragma once
#include <iostream>
#include <coroutine>
#include <ranges>
#include <memory>
#include <__generator.hpp>  //reference implementation of generator
#include <argparse.hpp>

#include <ATen/autocast_mode.h>
#include <torch/torch.h>
#include <sentencepiece_processor.h>

#include <rtg.hpp>
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

    enum Mode{
        TRAINING,
        INFERENCE,
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

        template <typename M>
        auto init_model(config::Config& config, torch::Device& device) -> M {
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

        template <typename C>
        auto init_criterion(const config::Config& config) -> C {
            return nn::CrossEntropyLoss(nn::CrossEntropyLossOptions().reduction(torch::kNone));
        }

        template <typename M>
        auto init_optimizer(const config::Config& config, M model)
             -> std::shared_ptr<optim::Optimizer> {
            auto optim_config = config["optimizer"];
            auto optim_name = optim_config["name"].as<string>();
            return std::make_shared<optim::Adam>(model->parameters(), optim::AdamOptions(0.0001));
        }

        auto init_scheduler(const config::Config& config, optim::Optimizer& optimizer)
             -> std::shared_ptr<optim::LRScheduler> {
            return std::make_shared<optim::StepLR>(optimizer, 1.0, 0.95);
        }

        auto init_config(fs::path work_dir, fs::path config_file) -> config::Config {
            /*
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
                fs::copy(config_file, work_config, fs::copy_options::overwrite_existing);
            }
            if (!fs::exists(work_config)) {
                throw runtime_error(fmt::format("Config file {} not found", work_config.string()));
            }
            return config::Config(config_file);
        }

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
        int32_t chunk_size;
    public:
        
        Trainer(fs::path work_dir, config::Config conf):
            work_dir {work_dir},
            config {conf},
            model {init_model<M>(config, device)},
            criterion {init_criterion<C>(config)},
            optimizer {init_optimizer(config, model)},
            scheduler {init_scheduler(config, *optimizer)},
            vocabs {load_vocabs(config)},
            chunk_size {config["trainer"]["chunk_size"].as<int>(-1)}
        {
            spdlog::info("Trainer initialized; work_dir={} device = {}", work_dir, device == torch::kCUDA ? "cuda" : "cpu");
        }

        Trainer(fs::path work_dir, fs::path config_file)
        : Trainer(work_dir, init_config(work_dir, config_file))
        {}

        ~Trainer() {}

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

        auto subsequent_mask(int64_t seq_len, torch::Device device=torch::kCPU) -> Tensor{
            // batch: [batch_size, seq_len]
            // pad_idx: padding token id; usually 0; ignore if -1
            // returns: [batch_size, seq_len, seq_len]
            auto mask = torch::ones({seq_len, seq_len}, torch::dtype(torch::kInt8).device(device)); // all cells have 1
            mask = torch::triu(mask, /*diagonal=*/1);            // upper triangle and diagonal are 1, lower diagonal are 0
            return mask;
        }

        void train() {
            size_t num_epochs = config["trainer"]["epochs"].as<int>();
            auto model = this->model;
            model->train();
            model->to(device);
            LOG::info("Moving to device {}", device == torch::kCUDA ? "cuda" : "cpu");

            spdlog::info("Training started; total epochs = {}", num_epochs);
            size_t step_num = 0;
            size_t tot_sents = 0;
            size_t tot_tokens = 0;
            const bool fp16_enabled = config["trainer"]["fp16"].as<bool>(false);
            const int64_t pad_id = 0;
            if (fp16_enabled) {
                spdlog::info("FP16 training enabled");
            }
            nn::AnyModule projector(model->lm_head);
            for (int32_t epoch = 0; epoch < num_epochs; epoch++) {
                auto train_data = get_train_data();
                for (auto batch : train_data) {
                    if (fp16_enabled){  //__enter__()
                        at::autocast::set_enabled(true);
                    }
                    batch = batch.to(device);
                    auto src_ids = batch.fields[0];  // [batch_size, seq_len]
                    auto tgt_ids = batch.fields[1];   // [batch_size, seq_len]
                    auto src_mask = (src_ids == pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
                    auto tgt_mask_padding = (tgt_ids == pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len] 
                    auto tgt_mask_autoreg = subsequent_mask(tgt_ids.size(1), device).unsqueeze(0).unsqueeze(1); // [1, 1, seq_len, seq_len] 
                    auto tgt_mask = tgt_mask_padding | tgt_mask_autoreg.type_as(tgt_mask_padding); // [batch_size, 1, seq_len, seq_len]
                    auto output = model(src_ids, tgt_ids, src_mask, tgt_mask);
                    auto normalizer = (tgt_ids != pad_id).sum().item().toInt(); // #total - #mask
                    //auto loss = simple_loss_computer(output, tgt_ids, projector, pad_id);
                    auto loss = loss_computer(output, tgt_ids, projector, pad_id, normalizer, Mode::TRAINING);

                    if (fp16_enabled){  // __exit__()
                        at::autocast::clear_cache();
                        at::autocast::set_enabled(false);
                    }
                    //loss.backward();
                    optimizer->step();
                    scheduler->step();
                    optimizer->zero_grad();

                    tot_sents += src_ids.sizes()[0];
                    tot_tokens += normalizer;
                    if (step_num % 25 == 0) {
                        spdlog::info("Step: {}; loss: {:.5f}; sents: {}; toks: {}",
                                    step_num, loss.item().toFloat(), tot_sents, tot_tokens);
                        }
                    step_num++;
                }
            }
        }

        auto loss_computer(Tensor features, Tensor labels, nn::AnyModule projector,
         int64_t pad_id, int64_t normalizer=-1, Mode mode=Mode::TRAINING){

            if (chunk_size > 0){

                return chunked_loss_computer(features, labels, projector, pad_id, chunk_size, normalizer, mode);
            } else {
                return simple_loss_computer(features, labels, projector, pad_id, normalizer, mode);
            }
         }

        auto simple_loss_computer(Tensor features, Tensor labels, nn::AnyModule projector,
         int64_t pad_id, int64_t normalizer=-1, Mode mode=Mode::TRAINING) -> Tensor{
            auto output = projector.forward(features);
            auto output_flat = output.view({output.size(0) * output.size(1), -1}); // [batch_size * seq_len, vocab_size]
            auto labels_flat = labels.view({-1}); // [batch_size * seq_len]
            Tensor loss = criterion(output_flat, labels_flat);  // [batch_size * seq_len]
            //exclude padding tokens from loss calculation
            loss.masked_fill_(labels_flat == pad_id, 0.0);
            if (normalizer <= 0) { // self computed normalizer
                normalizer = (labels_flat != pad_id).sum().item().toInt(); // #total - #mask
            }
            loss = loss.sum() / normalizer;
            if (mode == Mode::TRAINING) {
                loss.backward();
            }
            return loss;
        }

        auto chunked_loss_computer(Tensor features, const Tensor labels, nn::AnyModule projector,
            const int64_t pad_id, const size_t chunk_size, const int64_t normalizer=-1, const Mode mode=Mode::TRAINING) -> Tensor{
            /**
             * Compute loss in chunks to avoid OOM
             * features: [batch_size, seq_len, hidden_size]
             * labels: [batch_size, seq_len]
             * projector: nn::AnyModule that projects hid_size to vocab_size
             * pad_id: labels to exclude from loss calculation
             * chunk_size: size of chunk to use for loss computation
             * normalizer: total number of tokens in batch. If not provided, it is computed based on pad_id
            */
            if (chunk_size <= 0) {
                throw runtime_error("chunk_size must be > 0");
            }
            const size_t seq_len = features.size(1);
            const auto total_chunks = ceil((1.0 * seq_len) / chunk_size);
            Tensor total_loss = torch::tensor(0.0, torch::device(features.device()).dtype(torch::kFloat32));
            total_loss.requires_grad_(false); // cant do backward on this loss value

            // disconnect graph, accum grad across chunks, and then do backward
            Tensor features_isolated = features.detach().clone();
            features_isolated.requires_grad_(true);
            for (auto chunk_idx = 0z; chunk_idx < total_chunks; chunk_idx++) {
                auto start = chunk_idx * chunk_size;
                auto end = min((chunk_idx + 1) * chunk_size, seq_len);
                auto chunk_features = features_isolated.index({Slice(), Slice(start, end), Slice()}).contiguous();
                auto chunk_labels = labels.index({Slice(), Slice(start, end)}).contiguous();
                auto chunk_loss = simple_loss_computer(chunk_features, chunk_labels, projector, pad_id, normalizer, mode);
                total_loss += chunk_loss.item().toFloat();
            }

            if (mode == Mode::TRAINING) {
                features.backward(features_isolated.grad().data());
            }
            return total_loss;
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


// AMP/fp16 is discussed here https://discuss.pytorch.org/t/deploy-mixed-precision-model-in-libtorch/89046/5 
