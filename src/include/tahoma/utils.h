#pragma once

#include <tahoma.h>
#include <tahoma/config.h>
#include <tahoma/train/stats_counter.h>
#include <tahoma/model.h>
#include <tahoma/train/criterion.h>
#include <tahoma/train/loss_computer.h>
#include <sentencepiece_processor.h>


using namespace tahoma;
namespace optim = torch::optim;

namespace tahoma::utils {

    auto init_model(config::Config& config, torch::Device& device) -> std::shared_ptr<model::LanguageModel>;

    auto restore_model(const std::string& model_path, torch::Device& device, bool validate_config=true) 
        -> std::pair<config::Config, std::shared_ptr<model::LanguageModel>>;
    auto load_checkpt(const std::string& model_path, bool validate_config) -> std::pair<config::Config, Pack>;

    auto init_criterion(const YAML::Node& config, i64 ignore_idx) -> nn::AnyModule;

    auto init_optimizer(const config::Config& config, std::shared_ptr<model::LanguageModel> model)
        -> std::shared_ptr<optim::Optimizer>;

    auto init_scheduler(const config::Config& config, optim::Optimizer& optimizer, i64 initial_step=0)
        -> std::shared_ptr<train::LRScheduler>;

    auto init_config(fs::path work_dir, fs::path config_file) -> config::Config;

    auto load_vocab(const std::string& vocab_path) -> std::shared_ptr<sentencepiece::SentencePieceProcessor>;

    auto load_vocabs(const std::vector<std::string> vocab_paths) -> std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>>;

    auto subsequent_mask(i64 seq_len, torch::Device device = torch::kCPU) -> Tensor;

    auto init_loss_computer(const config::Config& config, nn::AnyModule& projector, const i64 pad_id) -> std::shared_ptr<train::LossComputer>;

    auto ends_with(const std::string& str, const std::vector<std::string>& candidates) -> bool;

    auto read_file(const std::string& path) -> std::string;
    //auto read_lines(std::ifstream& stream) -> Generator<std::string>;
    auto read_lines(const std::string& path) -> Generator<std::string>;
    auto split(const std::string& text, const std::string& delimiter) -> std::vector<std::string>;

    auto debug_message(bool condition, const std::string& message, Tensor data) -> void;
    auto debug_message(bool condition, const std::string& message, Pack& data, std::initializer_list<string> keys) -> void;


    struct Timer {

        std::string name;
        std::chrono::time_point<std::chrono::high_resolution_clock> start;

        Timer(std::string name="")
        : name {name}, start {std::chrono::high_resolution_clock::now()}{
            spdlog::info("Timer {} started", name);
        }

        float elapsed() {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(now - start).count() / 1e6;
        }

        ~Timer() {
            spdlog::info("Timer {} ended: {:.3f}s", name, elapsed());
        }
    };


} // namespace tahoma::train

