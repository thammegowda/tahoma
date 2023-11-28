#pragma once
#include <iostream>
#include <coroutine>
#include <ranges>
#include <memory>
#include <chrono>
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
using namespace chrono;

namespace rtg::train {

    enum Mode {
        TRAINING,
        INFERENCE,
    };

    template <typename M>
    auto init_model(config::Config& config, torch::Device& device) -> M {
        auto model_type = config["model"]["name"].as<string>();
        if (model_type == "transformer") {
            YAML::Node model_args = config["model"]["args"];
            auto model = nmt::transformer::TransformerNMT(model_args);
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
        if (optim_name == "adam") {
            auto options = optim::AdamOptions(optim_config["lr"].as<double>(0.0001));
            if (optim_config["weight_decay"].IsDefined()) {
                options.weight_decay(optim_config["weight_decay"].as<double>());
            }
            if (optim_config["betas"].IsDefined()) {
                auto betas = optim_config["betas"].as<vector<double>>();
                options.betas({ betas[0], betas[1] });
            }
            if (optim_config["eps"].IsDefined()) {
                options.eps(optim_config["eps"].as<double>());
            }
            if (optim_config["amsgrad"].IsDefined()) {
                options.amsgrad(optim_config["amsgrad"].as<bool>());
            }
            LOG::info("Optimizer {}", optim_name);
            return std::make_shared<optim::Adam>(model->parameters(), options);
        } else {
            throw runtime_error("Unknown or unsupported optimizer " + optim_name);
        }
    }

    auto init_scheduler(const config::Config& config, optim::Optimizer& optimizer)
        -> std::shared_ptr<optim::LRScheduler> {
        auto scheduler_config = config["scheduler"];
        auto scheduler_name = scheduler_config["name"].as<string>();
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
            if (!fs::exists(work_dir)) {
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


    struct StatsCounter {
        int64_t step_num = 0;
        int64_t tot_sents = 0;
        int64_t tot_tokens = 0;
        double tot_loss = 0.0;

        string log_frequency = "";
        int64_t log_frequency_step = -1;
        int64_t log_frequency_tokens = -1;
        int64_t log_frequency_time_sec = -1;

        int64_t last_log_step = 0;
        int64_t last_log_tokens = 0;
        chrono::time_point<chrono::high_resolution_clock> last_log_time = chrono::high_resolution_clock::now();
        chrono::time_point<chrono::high_resolution_clock> start_time = chrono::high_resolution_clock::now();


        auto set_log_frequency(string arg){
            /*
             arg should be an integer with a suffix
            suffix u =>  number of steps i.e updates
            suffix t => number of tokens
            the second last character is k,M,B (optional), then it is interpreted as kilo, million, billion 
            suffix s,m,h =>  number of seconds, minutes, hours 
            */
            log_frequency = arg;
            assert(arg.size() > 0);
            auto suffix = arg.back();
            auto num = arg.substr(0, arg.size() - 1);
            int64_t scaler = 1;
            // check if the second last character is k,M,B
            if (suffix == 'u' || suffix == 't'){
                switch (num.size() > 0 ? num[num.size() - 1] : '?') {
                    case 'k':  // proper
                    case 'K': // typo
                        scaler = 1'000; 
                        break;
                    case 'M': // proper
                        scaler = 1'000'000; 
                        break;
                    case 'B': // proper
                    case 'G': // okay
                        scaler = 1'000'000'000;
                        break;
                }
                if (scaler > 1) {
                    num = num.substr(0, num.size() - 1);
                }
            }

            size_t num_val = std::stoi(num);
            switch (suffix) {
                case 'u': // updates
                    log_frequency_step = scaler * num_val;
                    break;
                  case 't': // tokens
                    log_frequency_tokens = scaler * num_val;
                    break;
                case 's': // seconds
                    log_frequency_time_sec =  num_val;
                    break;
                case 'm': // minutes
                    log_frequency_time_sec = num_val * 60;
                    break;
                case 'h': // hours
                    log_frequency_time_sec = num_val * 60 * 60;
                    break;
                default:
                    throw runtime_error(fmt::format("Invalid log frequency argument {}", arg));
            }
        }

        // add constructor
        StatsCounter() {}
        StatsCounter(string log_frequency) {
            set_log_frequency(log_frequency);
        }

        // copy and move
        StatsCounter(const StatsCounter& other) = default;
        StatsCounter(StatsCounter&& other) = default;
        StatsCounter& operator=(const StatsCounter& other) = default;
        StatsCounter& operator=(StatsCounter&& other) = default;
        
        double avg_loss() {
            return step_num > 0 ? tot_loss / step_num : 0.0;
        }

        auto update(double loss, size_t num_sents, size_t num_tokens, size_t num_steps=1) -> StatsCounter& {
            tot_sents += num_sents;
            tot_tokens += num_tokens;
            step_num += num_steps;
            tot_loss += loss;
            bool log_now = false;
            if (log_frequency_step > 0 && step_num - last_log_step >= log_frequency_step) {
                log_now = true;
                last_log_step = step_num;
            } else if (log_frequency_tokens > 0 && num_tokens - last_log_tokens >= log_frequency_tokens) {
                log_now = true;
                last_log_tokens = num_tokens;
            } else if (log_frequency_time_sec > 0 && 
                chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - last_log_time).count() >= log_frequency_time_sec) {
                log_now = true;
                last_log_time = chrono::high_resolution_clock::now();
            }
            if (log_now) {
                auto duration_ms = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time);
                auto toks_rate = 1000.0f * tot_tokens / duration_ms.count() ;
                auto sents_rate = 1000.0f * tot_sents / duration_ms.count();
                spdlog::info("Step: {}; Loss: {:.5f}; AvgLoss: {:.5f}; sents: {}; toks: {}, speed: {:.1f} tok/s {:.1f} sent/s", 
                    step_num, loss, avg_loss(),  tot_sents, tot_tokens, toks_rate, sents_rate);
            }
            return *this;
        }
    };


     struct Stopper {
        int64_t patience = 10;
        int64_t num_stalls = 0;
        double best_loss = numeric_limits<double>::infinity();
        Stopper(int64_t patience) : patience{ patience } {}

        int stop(double loss) {
            /*
                -1 => no stop; continue training
                0 => new best loss; maybe take a checkpoint
                1 => stop training; we have exhausted patience
            */
            if (loss <= best_loss) {
                best_loss = loss;
                num_stalls = 0;
                spdlog::info("New best loss: {:.5f}", loss);
                return 0;
            } else {
                num_stalls++;
                spdlog::info("No improvement in last {} validations; patience={}; best={:.5f}; current={:.5f}", num_stalls, patience, best_loss, loss);
                return num_stalls >= patience ? 1 : -1; 
            }
            return num_stalls >= patience;
            
        }
    };


    template <typename M, typename C>
    class Trainer {
    protected:
        fs::path work_dir;
        config::Config config;
        M model;
        C criterion;
        nn::AnyModule projector;
        std::shared_ptr<optim::Optimizer> optimizer;
        std::shared_ptr<optim::LRScheduler> scheduler;
        rtg::data::DataLoader data_loader;
        int32_t chunk_size;
        bool fp16_enabled;
        int64_t pad_id = 0;
        int64_t bos_id = 1;
        torch::Device device{ torch::cuda::is_available() ? "cuda" : "cpu" };

    public:
        Trainer(fs::path work_dir, config::Config conf)
            : work_dir{ work_dir },
            config{ conf },
            model{ init_model<M>(config, device) },
            criterion{ init_criterion<C>(config) },
            projector{ model->lm_head },
            optimizer{ init_optimizer(config, model) },
            scheduler{ init_scheduler(config, *optimizer) },
            data_loader{ rtg::data::DataLoader(config) },
            chunk_size{ config["trainer"]["chunk_size"].as<int>(-1) },
            fp16_enabled{ config["trainer"]["fp16"].as<bool>(false) },
            pad_id {data_loader.vocabs[1]->pad_id()},   //vocabs[1] is the target vocab
            bos_id {data_loader.vocabs[1]->bos_id()}
        {
            spdlog::info("Trainer initialized; work_dir={} device={}; fp16={}", work_dir, device == torch::kCUDA ? "cuda" : "cpu", fp16_enabled);
        }

        Trainer(fs::path work_dir, fs::path config_file)
            : Trainer(work_dir, init_config(work_dir, config_file))
        {}

        ~Trainer() {}

        auto subsequent_mask(int64_t seq_len, torch::Device device = torch::kCPU) -> Tensor {
            // batch: [batch_size, seq_len]
            // pad_idx: padding token id; usually 0; ignore if -1
            // returns: [batch_size, seq_len, seq_len]
            auto mask = torch::ones({ seq_len, seq_len }, torch::dtype(torch::kInt8).device(device)); // all cells have 1
            mask = torch::triu(mask, /*diagonal=*/1);            // upper triangle and diagonal are 1, lower diagonal are 0
            return mask;
        }

        auto save_checkpoint(string tag = "") {
            auto filename = fmt::format("model{}.pt", tag.size() > 0 ? "." + tag : "");
            auto checkpoint_file = work_dir / filename;
            spdlog::info("Saving checkpoint to {}", checkpoint_file);
            torch::save(model, checkpoint_file.string());
            spdlog::info("Checkpoint saved");
        }

        auto step(data::Batch& batch, StatsCounter& stats, const Mode mode = Mode::TRAINING) -> Tensor {
             if (fp16_enabled) {  //__enter__()
                at::autocast::set_enabled(true);
            }
            batch.contiguous();
            batch = batch.to(device);
            auto src_ids = batch.fields[0];  // [batch_size, seq_len]
            auto tgt_ids = batch.fields[1];   // [batch_size, seq_len]
            //FIME: add bos and eos tokens to tgt_ids. Offset tgt_ids by 1
            auto _bos_col = torch::full({ tgt_ids.size(0), 1 }, bos_id, torch::dtype(torch::kInt64).device(device));
            tgt_ids = torch::cat({_bos_col, tgt_ids}, 1);
            
            auto src_mask = (src_ids == pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
            auto tgt_mask_padding = (tgt_ids == pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len] 
            auto tgt_mask_autoreg = subsequent_mask(tgt_ids.size(1), device).unsqueeze(0).unsqueeze(1); // [1, 1, seq_len, seq_len] 
            auto tgt_mask = tgt_mask_padding | tgt_mask_autoreg.type_as(tgt_mask_padding); // [batch_size, 1, seq_len, seq_len]
            auto normalizer = (tgt_ids != pad_id).sum().item().toInt(); // #total - #mask
            auto output = model(src_ids, tgt_ids, src_mask, tgt_mask);
            auto loss = loss_computer(output, tgt_ids, projector, pad_id, normalizer, mode);
            stats.update(loss.item().toDouble(), tgt_ids.size(0), normalizer);

            if (fp16_enabled) {  // __exit__()
                at::autocast::clear_cache();
                at::autocast::set_enabled(false);
            }
            if (mode == Mode::TRAINING){
                optimizer->step();
                scheduler->step();
                optimizer->zero_grad();
            }
            return loss;
        }

       
        void train() {
            LOG::info("Moving to device {}", device == torch::kCUDA ? "cuda" : "cpu");
            model->to(device);
            size_t num_epochs = config["trainer"]["epochs"].as<int>();
            auto log_frequency = config["trainer"]["log_frequency"].as<string>("25u");
            auto checkpoint_frequency = config["checkpoint"]["frequency"].as<size_t>(1000);
            auto validation_frequency = config["validator"]["frequency"].as<size_t>(1000);
            spdlog::info("Training started; total epochs = {}; log_frequency={}", num_epochs, log_frequency);
            model->train();

            Stopper stopper(config["trainer"]["early_stopping"].as<int>(8));

            for (int32_t epoch = 0; epoch < num_epochs; epoch++) {
                auto stats = StatsCounter(log_frequency);
                for (auto batch : data_loader.get_train_data()) {
                    step(batch, stats, Mode::TRAINING);
                    batch.fields.clear(); // clear references to tensors

                    if (stats.step_num % validation_frequency == 0) {
                        {
                            torch::AutoGradMode enable_grad(false);
                            model->eval();
                            auto valid_loss = validate();
                            switch (stopper.stop(valid_loss)) {
                                case 0: // new best loss
                                    save_checkpoint("best");
                                    break;  
                                case 1: // stop training
                                    LOG::info("Early stopping at epoch {} step {}", epoch, stats.step_num);
                                    return;
                                case -1: // continue training
                                    break;
                                default:
                                    throw runtime_error("Invalid stopper return value");
                            }
                        }
                        model->train();
                    }
                    if (stats.step_num % checkpoint_frequency == 0) {
                        save_checkpoint(fmt::format("step.{}", stats.step_num));
                    }
                }
                // todo validation
            }
        }

        auto validate(){
            auto stats = StatsCounter(config["trainer"]["log_frequency"].as<string>("25u"));
            LOG::info("Starting validation. log_frequency={}", stats.log_frequency);
            for (auto batch :  data_loader.get_validation_data()) {
                step(batch, stats, Mode::INFERENCE);
            }
            LOG::info("Validation finished. Avg loss: {:.5f}", stats.avg_loss());
            return stats.avg_loss();
        }

    
        auto loss_computer(Tensor features, Tensor labels, nn::AnyModule projector,
            int64_t pad_id, int64_t normalizer = -1, Mode mode = Mode::TRAINING) {

            if (chunk_size > 0) {
                return chunked_loss_computer(features, labels, projector, pad_id, chunk_size, normalizer, mode);
            } else {
                return simple_loss_computer(features, labels, projector, pad_id, normalizer, mode);
            }
        }

        auto simple_loss_computer(Tensor features, Tensor labels, nn::AnyModule projector,
            int64_t pad_id, int64_t normalizer = -1, Mode mode = Mode::TRAINING) -> Tensor {
            auto output = projector.forward(features);
            auto output_flat = output.view({ output.size(0) * output.size(1), -1 }); // [batch_size * seq_len, vocab_size]
            auto labels_flat = labels.reshape({ -1 }); // [batch_size * seq_len]
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
            const int64_t pad_id, const size_t chunk_size, const int64_t normalizer = -1,
            const Mode mode = Mode::TRAINING) -> Tensor {
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

            // disconnect graph, ulate grad across chunks, and then do backward
            Tensor features_isolated = features.detach().clone();
            features_isolated.requires_grad_(true);
            for (auto chunk_idx = 0z; chunk_idx < total_chunks; chunk_idx++) {
                auto start = chunk_idx * chunk_size;
                auto end = min((chunk_idx + 1) * chunk_size, seq_len);
                auto chunk_features = features_isolated.index({ Slice(), Slice(start, end), Slice() });
                auto chunk_labels = labels.index({ Slice(), Slice(start, end) });
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
    spdlog::info("main started.. torch version: {} ", TORCH_VERSION);
    auto args = train::parse_args(argc, argv);
    if (args.get<bool>("verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }
    auto work_dir = fs::path{ args.get<std::string>("work_dir") };
    auto config_file_arg = args.get<std::string>("config");
    fs::path config_file;
    if (!config_file_arg.empty()) {
        config_file = fs::path{ config_file_arg };
    }
    auto trainer = train::Trainer<nmt::transformer::TransformerNMT, nn::CrossEntropyLoss>(work_dir, config_file);
    trainer.train();
    spdlog::info("main finished..");
    return 0;
}


// AMP/fp16 is discussed here https://discuss.pytorch.org/t/deploy-mixed-precision-model-in-libtorch/89046/5 
