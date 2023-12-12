#pragma once
#include <iostream>
#include <coroutine>
#include <ranges>
#include <memory>
#include <__generator.hpp>  //reference implementation of generator

#include <torch/torch.h>
#include <sentencepiece_processor.h>

#include <rtg.hpp>
#include "../common/config.hpp"
#include "../common/data.hpp"
#include "./loss_computer.hpp"
#include "./criterion.hpp"



namespace nn = torch::nn;
namespace optim = torch::optim;
namespace sp = sentencepiece;
namespace fs = std::filesystem;

using namespace std;
using namespace torch::indexing;
using namespace rtg;

namespace rtg::train {

    template <typename M>
    auto init_model(config::Config& config, torch::Device& device) -> M {
        auto model_type = config["model"]["name"].as<string>();
        if (model_type == "transformer") {
            YAML::Node model_args = config["model"]["args"];
            auto model = nmt::transformer::TransformerNMT(model_args);
            return model;
        }
        else {
            throw runtime_error("Unknown model type " + model_type);
        }
    }

    auto init_criterion(const YAML::Node& config, i64 ignore_idx) -> train::CrossEntropyLoss {
        auto name = config["name"].as<string>("cross_entropy");
        if (name != "cross_entropy"){
            throw runtime_error("Unknown criterion " + name +". only cross_entropy supported");
        }
        f32 label_smooth_rate = config["args"]["label_smooth_rate"].as<f32>(0.0);
        auto criterion = train::CrossEntropyLoss(ignore_idx, label_smooth_rate);
        return criterion;
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
        }
        else {
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


    auto subsequent_mask(int64_t seq_len, torch::Device device = torch::kCPU) -> Tensor {
        // batch: [batch_size, seq_len]
        // pad_idx: padding token id; usually 0; ignore if -1
        // returns: [batch_size, seq_len, seq_len]
        auto mask = torch::ones({ seq_len, seq_len }, torch::dtype(torch::kInt8).device(device)); // all cells have 1
        mask = torch::triu(mask, /*diagonal=*/1);            // upper triangle and diagonal are 1, lower diagonal are 0
        return mask;
    }

    auto init_loss_computer(const config::Config& config, nn::AnyModule& projector, const i64 pad_id) -> shared_ptr<LossComputer> {
        auto trainer_criterion = init_criterion(config["trainer"]["criterion"], pad_id);
        map<string, train::CrossEntropyLoss> validation_criteria;
        for (auto criterion_config : config["validator"]["criteria"]) {
            auto name = criterion_config["name"].as<string>();
            validation_criteria[name] = init_criterion(criterion_config, pad_id);
        }
        auto chunk_size = config["trainer"]["chunk_size"].as<size_t>(0);
        auto container = make_shared<CriteriaContainer>(trainer_criterion, validation_criteria );
        return make_shared<LossComputer>(projector, container, pad_id, chunk_size);
    }


    enum StopperStatus {
        STOP,  // early stop reached
        NO_STOP, // continue training
        NEW_BEST, // new best loss, and continue training
    };

    struct Stopper {

        int32_t patience = 10;
        int32_t num_stalls = 0;
        float best_loss = numeric_limits<float>::infinity();
        Stopper(int32_t patience) : patience{ patience } {}

        auto is_stop(float loss) -> StopperStatus {
            using enum StopperStatus;
            if (loss <= best_loss) {
                best_loss = loss;
                num_stalls = 0;
                spdlog::info("New best loss: {:.5f}", loss);
                return NEW_BEST;
            }
            else {
                num_stalls++;
                spdlog::info("No improvement in last {} validations; patience={}; best={:.5f}; current={:.5f}", num_stalls, patience, best_loss, loss);
                return num_stalls >= patience ? STOP : NO_STOP;
            }
        }
    };

} // namespace rtg::train

