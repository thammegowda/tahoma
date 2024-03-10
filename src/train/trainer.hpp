#pragma once
#include <iostream>
#include <coroutine>
#include <ranges>
#include <vector>
#include <memory>
#include <chrono>
#include <__generator.hpp>  //reference implementation of generator

#include <ATen/autocast_mode.h>
#include <torch/torch.h>
#include <sentencepiece_processor.h>

#include <tahoma.hpp>
#include "../common/config.hpp"
#include "../common/data.hpp"
#include "../model/transformer_nmt.hpp"
#include "../model/transformer_lm.hpp"
#include "./utils.hpp"
#include "./stats_counter.hpp"
#include "../inference/decoder.hpp"

namespace nn = torch::nn;
namespace optim = torch::optim;
namespace sp = sentencepiece;
namespace fs = std::filesystem;

using namespace std;
using namespace torch::indexing;
using namespace tahoma;
using namespace chrono;

namespace tahoma::train {

    auto DEVICE = torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    //template <typename M = model::LanguageModel>
    class Trainer {
    protected:
        fs::path _work_dir;
        config::Config _config;
        std::shared_ptr<model::LanguageModel> _model;
        //nn::AnyModule _model;
        nn::AnyModule _projector;
        shared_ptr<optim::Optimizer> _optimizer;
        shared_ptr<train::LRScheduler> _scheduler;
        tahoma::data::DataLoader _data_loader;

        bool _fp16_enabled;
        int64_t _pad_id = 0;
        int64_t _bos_id = 1;
        shared_ptr<LossComputer> _loss_computer;
        torch::Device _device;
        data::Batch _sample_batch;
        Stopper _stopper;

    public:
        Trainer(fs::path work_dir, config::Config conf)
            : _work_dir{ work_dir },
            _config{ conf },
            _device{ DEVICE },
            _model{ init_model(_config, _device)},
            _projector{ _model->lm_head },
            _optimizer{ init_optimizer(_config, _model) },
            _scheduler{ init_scheduler(_config, *_optimizer) },

            _data_loader{ tahoma::data::DataLoader(_config) },
            _fp16_enabled{ _config["trainer"]["fp16"].as<bool>(false) },
            _pad_id {_data_loader.vocabs[1]->pad_id()},   //vocabs[1] is the target vocab
            _bos_id {_data_loader.vocabs[1]->bos_id()},
            _loss_computer{  init_loss_computer(_config, _projector, _pad_id) },
            _sample_batch{ _data_loader.get_samples(_config["validator"]["data"].as<vector<string>>(), /*n_samples*/5) },
            _stopper{ _config["trainer"]["early_stopping"].as<int>(8) }
        {
            spdlog::info("Trainer initialized; work_dir={}, cuda_available?={} device={}; fp16={}",
                work_dir, torch::cuda::is_available(), _device == torch::kCUDA ? "cuda" : "cpu", _fp16_enabled);
            spdlog::info("Early stopping enabled? {}, patience: {}",  _stopper.patience > 0 ? "Yes" : "No",  _stopper.patience);
            // check if pad_id is correct
            if (_pad_id < 0){
                throw std::runtime_error("pad_id is negative, implying it is disabled, however it is required. Please create a new vocab with pad_id.");
            }
        }

        Trainer(fs::path work_dir, fs::path config_file)
            : Trainer(work_dir, init_config(work_dir, config_file))
        {}

        ~Trainer() {}

        auto save_checkpoint(string tag = "") {
            auto filename = fmt::format("model{}.pt", tag.size() > 0 ? "." + tag : "");
            auto checkpoint_file = _work_dir / filename;
            spdlog::info("Saving checkpoint to {}", checkpoint_file);
            torch::save(_model, checkpoint_file.string());
        }

        auto step(data::Batch& batch, StatsCounter& stats, const Mode mode = Mode::TRAINING) -> Tensor {
            torch::AutoGradMode enable_grad(mode == Mode::TRAINING); // RAII
             if (_fp16_enabled) {  //__enter__()
                at::autocast::set_enabled(true);
            }
            if (mode == Mode::TRAINING){
                _optimizer->zero_grad();
            }
            batch.contiguous();
            batch = batch.to(_device);
            auto src_ids = batch.fields[0];  // [batch_size, seq_len]
            auto tgt_ids = batch.fields[1];   // [batch_size, seq_len]
            //FIME: add bos and eos tokens to tgt_ids. Offset tgt_ids by 1
            auto _bos_col = torch::full({ tgt_ids.size(0), 1 }, _bos_id, torch::dtype(torch::kInt64).device(_device));
            auto bos_tgt_ids = torch::cat({_bos_col, tgt_ids}, 1);

            auto src_mask = src_ids.eq(_pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
            auto tgt_mask_padding = bos_tgt_ids.eq(_pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
            auto tgt_mask_autoreg = subsequent_mask(bos_tgt_ids.size(1), _device).unsqueeze(0).unsqueeze(1); // [1, 1, seq_len, seq_len]
            auto tgt_mask = tgt_mask_padding | tgt_mask_autoreg.type_as(tgt_mask_padding); // [batch_size, 1, seq_len, seq_len]
            auto normalizer = (bos_tgt_ids != _pad_id).sum().item().toInt(); // #total - #mask

            bool debug_input = false;
            if (debug_input) {
                // print batch for debugging
                for (auto i=0; i <batch.examples.size(); i++){
                    auto ex = batch.examples[i];
                    str src = ex.fields[0];
                    str ref = ex.fields[1];
                    std::cout << "SRC: " << src << std::endl;
                    std::cout << "REF: " << ref << std::endl;
                }
                std::cout << "src_ids: " << src_ids << std::endl;
                std::cout << "bos_tgt_ids: " << bos_tgt_ids << std::endl;
                std::cout << "pad_id: " << _pad_id << std::endl;
                std::cout << "src_mask: " << src_mask << std::endl;
                std::cout << "tgt_mask: " << tgt_mask << std::endl;
                std::cout << "normalizer: " << normalizer << std::endl;
            }

            //auto features = _model->forward(src_ids, src_mask, bos_tgt_ids, tgt_mask);
            Pack pack = {
                {"src", src_ids},
                {"src_mask", src_mask},
                {"tgt", bos_tgt_ids},
                {"tgt_mask", tgt_mask}
            };
            auto result = _model->forward(pack);
            auto features = std::any_cast<Tensor>(result["result"]); // [batch_size, seq_len, model_dim]
            // skip the last token (EOS) in features, as it is not used in loss computation
            features = features.index({ Slice(), Slice(0, -1), Slice() }); // [batch_size, seq_len, model_dim]
            auto loss = _loss_computer->compute(features, tgt_ids, normalizer, mode);

            if (_fp16_enabled) {  // __exit__()
                at::autocast::clear_cache();
                at::autocast::set_enabled(false);
            }
            if (mode == Mode::TRAINING){
                //torch::nn::utils::clip_grad_norm_(_model->parameters(), _config["trainer"]["clip_grad"].as<f32>(5.0));
                _optimizer->step();
                _scheduler->step();
            }
            stats.update(loss.item().toDouble(), tgt_ids.size(0), normalizer, _scheduler->get_last_rate());
            return loss;
        }


        void train() {
            LOG::info("Moving to device {}", _device == torch::kCUDA ? "cuda" : "cpu");
            _model->to(_device);
            size_t num_epochs = _config["trainer"]["epochs"].as<int>();
            auto log_frequency = _config["trainer"]["log_frequency"].as<string>("25u");
            auto checkpoint_frequency = _config["checkpoint"]["frequency"].as<size_t>(1000);
            auto validation_frequency = _config["validator"]["frequency"].as<size_t>(1000);
            spdlog::info("Training started; total epochs = {}; log_frequency={}", num_epochs, log_frequency);
            _model->train();

            Stopper stopper(_config["trainer"]["early_stopping"].as<int>(8));
            int16_t log_first = _config["trainer"]["log_first"].as<int16_t>(0);
            auto stats = StatsCounter(log_frequency, /*name=*/"Training", /*log_first=*/log_first);
            for (i32 epoch = 0; epoch < num_epochs; epoch++) {
                for (auto batch : _data_loader.get_train_data()) {
                    step(batch, stats, Mode::TRAINING);
                    batch.fields.clear(); // clear references to tensors

                    if (stats.step_num % validation_frequency == 0) {
                        bool is_early_stop = validate();
                        if (is_early_stop) {
                            LOG::info("Early stopping at epoch {} step {}", epoch, stats.step_num);
                            // todo: throw an EarlyStopException
                            return;
                        }
                        _model->train();
                    }
                    if (stats.step_num % checkpoint_frequency == 0) {
                        save_checkpoint(fmt::format("step_{}", stats.step_num));
                    }
                }
            }
        }

        void log_samples(){
            std::shared_ptr<model::TransformerNMTImpl> model = std::dynamic_pointer_cast<model::TransformerNMTImpl>(_model);
            auto decoder = inference::Decoder(model, _projector, _data_loader.vocabs, _device);
            for (auto i=0; i < _sample_batch.size(); i++) {
                auto ex = _sample_batch.examples[i];
                str src = ex.fields[0];
                str ref = ex.fields[1];
                spdlog::info("[{}] SRC: {}", i, src);
                spdlog::info("[{}] REF: {}", i, ref);
                auto [hyp, score] = decoder.greedy_decode(src);
                spdlog::info("[{}] HYP: ({}) {}", i, score, hyp);
            }
        }

        auto validate(bool show_samples=true) -> bool {
            /**
             * Returns true if early stopping is triggered
            */
            torch::AutoGradMode enable_grad(false);
            _model->eval();
            if (show_samples) {
                log_samples();
            }
            auto log_frequency = _config["validator"]["log_frequency"].as<string>("5000u");
            auto stats = StatsCounter(log_frequency, /*name*/"Validation");
            LOG::info("Starting validation. log_frequency={}", stats.log_frequency);
            for (auto batch : _data_loader.get_validation_data()) {
                step(batch, stats, Mode::INFERENCE);
            }
            LOG::info("Validation finished. Avg loss: {:.5f}", stats.avg_loss());
            f32 valid_loss = stats.avg_loss();
            // TODO: support multiple validation metrics

            switch (_stopper.is_stop(valid_loss)) {
                case StopperStatus::STOP: // stop training
                    save_checkpoint("last");
                    return true;
                case StopperStatus::NEW_BEST: // new best loss
                    spdlog::info("Saving new best model");
                    save_checkpoint("best");
                    return false;
                case StopperStatus::CONTINUE: // continue training
                    spdlog::info("No improvement in validation metric. Number of stalls: {}, (patience: {}) ", _stopper.num_stalls, _stopper.patience);
                default:
                    return false;
            }
        }
    };

}


// AMP/fp16 is discussed here https://discuss.pytorch.org/t/deploy-mixed-precision-model-in-libtorch/89046/5 
