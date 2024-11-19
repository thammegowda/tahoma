#pragma once

#include <tahoma.h>
#include <tahoma/config.h>
#include <tahoma/train/stats_counter.h>
#include <tahoma/model.h>
#include <tahoma/train/criterion.h>
#include <tahoma/train/loss_computer.h>


using namespace tahoma;
namespace optim = torch::optim;

namespace tahoma::train {

    auto init_model(config::Config& config, torch::Device& device) -> std::shared_ptr<model::LanguageModel>;

    auto init_criterion(const YAML::Node& config, i64 ignore_idx) -> nn::AnyModule;

    auto init_optimizer(const config::Config& config, std::shared_ptr<model::LanguageModel> model)
        -> std::shared_ptr<optim::Optimizer>;

    auto init_scheduler(const config::Config& config, optim::Optimizer& optimizer, i64 initial_step=0)
        -> std::shared_ptr<train::LRScheduler>;

    auto init_config(fs::path work_dir, fs::path config_file) -> config::Config;

    auto subsequent_mask(i64 seq_len, torch::Device device = torch::kCPU) -> Tensor;

    auto init_loss_computer(const config::Config& config, nn::AnyModule& projector, const i64 pad_id) -> std::shared_ptr<LossComputer>;

    enum class StopperStatus {
        STOP,  // early stop reached
        CONTINUE, // continue training
        NEW_BEST, // new best loss, and continue training
    };

    struct Stopper {
        int32_t patience = 10;
        int32_t num_stalls = 0;
        float best_loss = std::numeric_limits<float>::infinity();
        Stopper(int32_t patience) : patience{ patience } {}
        auto is_stop(float loss) -> StopperStatus;
    };

} // namespace tahoma::train

