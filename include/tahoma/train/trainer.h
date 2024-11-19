#pragma once
#include <tahoma.h>
#include <tahoma/train/stats_counter.h>
#include <tahoma/train/criterion.h>
#include <tahoma/train/loss_computer.h>
#include <tahoma/train/utils.h>
#include <tahoma/model.h>
#include <tahoma/data.h>




namespace tahoma::train {
    class Trainer {
    protected:
        fs::path _work_dir;
        config::Config _config;
        std::shared_ptr<model::LanguageModel> _model;
        TaskType _task_type;
        //nn::AnyModule _model;
        nn::AnyModule _projector;
        std::shared_ptr<optim::Optimizer> _optimizer;
        std::shared_ptr<LRScheduler> _scheduler;
        tahoma::data::DataLoader _data_loader;

        bool _fp16_enabled;
        int64_t _pad_id = 0;
        int64_t _bos_id = 1;
        std::shared_ptr<LossComputer> _loss_computer;
        torch::Device _device;
        data::Batch _sample_batch;
        Stopper _stopper;

        auto step_nmt(data::Batch& batch, StatsCounter& stats, const Mode mode = Mode::TRAINING) -> Pack;
        auto step_lm(data::Batch& batch, StatsCounter& stats, const Mode mode = Mode::TRAINING) -> Pack;

    public:
        Trainer(fs::path work_dir, config::Config conf);
        Trainer(fs::path work_dir, fs::path config_file);
        ~Trainer() = default;

        void save_checkpoint(std::string tag = "");
        auto step(data::Batch& batch, StatsCounter& stats, const Mode mode = Mode::TRAINING) -> Tensor;
        void train();
        void log_nmt_samples();
        auto validate(bool show_samples=true) -> bool;
    };
}