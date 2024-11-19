#include <iostream>
#include <coroutine>
#include <ranges>
#include <memory>

#include <torch/torch.h>
#include <sentencepiece_processor.h>

#include <tahoma.h>
#include <tahoma/model/transformer_nmt.h>
#include <tahoma/model/transformer_lm.h>
#include <tahoma/train/stats_counter.h>
#include <tahoma/train/criterion.h>
#include <tahoma/train/loss_computer.h>
#include <tahoma/train/utils.h>


using namespace tahoma;

namespace tahoma::train {

    auto init_model(config::Config& config, torch::Device& device) -> std::shared_ptr<model::LanguageModel> {
        auto model_type = config["model"]["name"].as<std::string>();
        YAML::Node model_args = config["model"]["args"];
        std::shared_ptr<model::LanguageModel> model;
        if (model_type == "transformer_nmt") {
            model = std::make_shared<model::TransformerNMTImpl>(model_args);
        } else if (model_type == "transformer_lm") {
            model = std::make_shared<model::TransformerLMImpl>(model_args);
        } else {
            throw std::runtime_error("Unknown model type " + model_type);
        }
        // NOTE: trying to move model to device here causes error. Not sure why.
        //LOG::info("Device: {}", device == torch::kCPU ? "CPU" : "CUDA");
        //model->to(device);
        return model;
    }

    auto init_criterion(const YAML::Node& config, i64 ignore_idx) -> nn::AnyModule {
        auto name = config["name"].as<std::string>("cross_entropy");
        if (name == "cross_entropy") {
           f32 label_smooth_rate = config["args"]["label_smooth_rate"].as<f32>(0.0);
           auto criterion = train::CrossEntropyLoss(ignore_idx, label_smooth_rate);
           return nn::AnyModule(criterion);
        } else if (name  == "kl_divergence" ) {
            f32 label_smooth_rate = config["args"]["label_smooth_rate"].as<f32>(0.0);
            i64 num_labels = config["args"]["num_labels"].as<i64>(0);
            if (num_labels < 1) {
                throw std::runtime_error("num_labels must be > 0 for kl_divergence with label_smoothing");
            }
            auto criterion = train::KLDivergence(num_labels, ignore_idx, label_smooth_rate);
           return nn::AnyModule(criterion);
        } else {
            throw std::runtime_error("Unknown criterion " + name +". only cross_entropy supported");
        }
    }

    //template <typename M>
    auto init_optimizer(const config::Config& config, /*nn::AnyModule*/ std::shared_ptr<model::LanguageModel> model)
        -> std::shared_ptr<optim::Optimizer> {
        auto optim_config = config["optimizer"];
        auto optim_name = optim_config["name"].as<std::string>();
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
            spdlog::info("Optimizer {}", optim_name);
            return std::make_shared<optim::Adam>(model->parameters(), options);
        } else {
            throw std::runtime_error("Unknown or unsupported optimizer " + optim_name);
        }
    }

    auto init_scheduler(const config::Config& config, optim::Optimizer& optimizer, i64 initial_step)
        -> std::shared_ptr<train::LRScheduler> {
        i64 start_step = 0; // TODO: restore from checkpt dir tor resume training
        auto scheduler_config = config["scheduler"];
        auto name = scheduler_config["name"].as<std::string>();
        YAML::Node options = scheduler_config["args"];
        if (name == "inverse_sqrt") {
            return std::make_shared<train::InverseSqrtScheduler>(optimizer, start_step, options);
        } else if (name == "noam") {
            return std::make_shared<train::NoamScheduler>(optimizer, start_step, options);
        } else {
            throw std::runtime_error("Unknown or unsupported scheduler " + name);
        }
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
                throw std::runtime_error(fmt::format("Config file {} not found", config_file.string()));
            }
            if (!fs::exists(work_dir)) {
                spdlog::info("mkdir {}", work_dir);
                fs::create_directories(work_dir);
            }
            spdlog::info("Copy {} ➡️ {}", config_file, work_config);
            fs::copy(config_file, work_config, fs::copy_options::overwrite_existing);
        }
        if (!fs::exists(work_config)) {
            throw std::runtime_error(fmt::format("Config file {} not found", work_config.string()));
        }
        return config::Config(config_file);
    }


    auto subsequent_mask(i64 seq_len, torch::Device device) -> Tensor {
        // input: seq_len
        // pad_idx: padding token id; usually 0; ignore if -1
        // returns: [seq_len, seq_len]
        auto mask = torch::ones({ seq_len, seq_len }, torch::dtype(torch::kInt8).device(device)); // all cells have 1
        mask = torch::triu(mask, /*diagonal=*/1);            // upper triangle and diagonal are 1, lower diagonal are 0
        return mask;
    }

    auto init_loss_computer(const config::Config& config, nn::AnyModule& projector, const i64 pad_id) -> std::shared_ptr<LossComputer> {
        auto trainer_criterion = init_criterion(config["trainer"]["criterion"], pad_id);
        std::map<std::string, nn::AnyModule> validation_criteria;
        for (auto criterion_config : config["validator"]["criteria"]) {
            auto name = criterion_config["name"].as<std::string>();
            validation_criteria[name] = init_criterion(criterion_config, pad_id);
        }
        auto chunk_size = config["trainer"]["chunk_size"].as<size_t>(0);
        auto container = std::make_shared<CriteriaContainer>(trainer_criterion, validation_criteria );
        return std::make_shared<LossComputer>(projector, container, pad_id, chunk_size);
    }

    auto Stopper::is_stop(float loss) -> StopperStatus {
        using enum StopperStatus;
        if (loss < best_loss) {
            best_loss = loss;
            num_stalls = 0;
            spdlog::info("New best loss: {:.5f}", loss);
            return NEW_BEST;
        } else {
            num_stalls++;
            spdlog::info("No improvement in last {} validations; patience={}; best={:.5f}; current={:.5f}", num_stalls, patience, best_loss, loss);
            return num_stalls >= patience ? STOP : CONTINUE;
        }
    }

} // namespace tahoma::train

