
#include <coroutine>
#include <ATen/autocast_mode.h>
#include <torch/torch.h>
#include <sentencepiece_processor.h>

#include <tahoma.h>
#include <tahoma/data.h>
#include <tahoma/model.h>
#include <tahoma/model/transformer_nmt.h>
#include <tahoma/model/transformer_lm.h>
#include <tahoma/inference/decoder.h>
#include <tahoma/train/stats_counter.h>
#include <tahoma/train/criterion.h>
#include <tahoma/train/loss_computer.h>
#include <tahoma/train/utils.h>
#include <tahoma/train/trainer.h>

namespace nn = torch::nn;
namespace optim = torch::optim;
namespace sp = sentencepiece;
namespace fs = std::filesystem;

using namespace tahoma;

namespace tahoma::train {

    auto DEVICE = torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);


    Trainer::Trainer(fs::path work_dir, config::Config conf)
        : _work_dir{ work_dir },
        _config{ conf },
        _device{ DEVICE },
        _model{ init_model(_config, _device)},
        _task_type{ _model->task_type() },
        _projector{ _model->lm_head },
        _optimizer{ init_optimizer(_config, _model) },
        _scheduler{ init_scheduler(_config, *_optimizer) },

        _data_loader{ tahoma::data::DataLoader(_config) },
        _fp16_enabled{ _config["trainer"]["fp16"].as<bool>(false) },
        _pad_id {_data_loader.output_vocab()->pad_id()},
        _bos_id {_data_loader.output_vocab()->bos_id()},
        _loss_computer{  init_loss_computer(_config, _projector, _pad_id) },
        _sample_batch{ _data_loader.get_samples(_config["validator"]["data"].as<std::vector<std::string>>(), /*n_samples*/5) },
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

    Trainer::Trainer(fs::path work_dir, fs::path config_file)
        : Trainer(work_dir, init_config(work_dir, config_file))
    {}

        void Trainer::save_checkpoint(std::string tag) {
            auto filename = fmt::format("model{}.pt", tag.size() > 0 ? "." + tag : "");
            auto checkpoint_file = _work_dir / filename;
            spdlog::info("Saving checkpoint to {}", checkpoint_file);
            torch::save(_model, checkpoint_file.string());
        }


        auto Trainer::step_nmt(data::Batch& batch, StatsCounter& stats, const Mode mode) -> Pack {

            assert(batch.fields.size() == 2 && "fields vector must have exactly two elements");
            auto src_ids = batch.fields[0];  // [batch_size, seq_len]
            auto tgt_ids = batch.fields[1];   // [batch_size, seq_len]
            //FIME: add bos and eos tokens to tgt_ids. Offset tgt_ids by 1
            auto _bos_col = torch::full({ tgt_ids.size(0), 1 }, _bos_id, torch::dtype(torch::kInt64).device(_device));
            auto bos_tgt_ids = torch::cat({_bos_col, tgt_ids}, 1);

            auto src_mask = src_ids.eq(_pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
            auto tgt_mask_padding = bos_tgt_ids.eq(_pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
            auto tgt_mask_autoreg = subsequent_mask(bos_tgt_ids.size(1), _device).unsqueeze(0).unsqueeze(1); // [1, 1, seq_len, seq_len]
            auto tgt_mask = tgt_mask_padding | tgt_mask_autoreg.type_as(tgt_mask_padding); // [batch_size, 1, seq_len, seq_len]
            auto normalizer = (bos_tgt_ids != _pad_id).sum().item().toLong(); // #total - #mask

            bool debug_input = false;
            //auto features = _model->forward(src_ids, src_mask, bos_tgt_ids, tgt_mask);
            Pack pack = {
                {"src", src_ids},
                {"src_mask", src_mask},
                {"tgt", bos_tgt_ids},
                {"tgt_mask", tgt_mask},
                {"normalizer", normalizer},
                {"labels", tgt_ids}
            };
           return pack;
        }

        auto Trainer::step_lm(data::Batch& batch, StatsCounter& stats, const Mode mode) -> Pack {
            auto inp_ids = batch.fields[0];  // [batch_size, seq_len]
            //FIME: add bos and eos tokens to tgt_ids. Offset tgt_ids by 1
            auto _bos_col = torch::full({ inp_ids.size(0), 1 }, _bos_id, torch::dtype(torch::kInt64).device(_device));
            auto bos_tgt_ids = torch::cat({_bos_col, inp_ids}, 1);

            auto tgt_mask_padding = bos_tgt_ids.eq(_pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
            auto tgt_mask_autoreg = subsequent_mask(bos_tgt_ids.size(1), _device).unsqueeze(0).unsqueeze(1); // [1, 1, seq_len, seq_len]
            auto tgt_mask = tgt_mask_padding | tgt_mask_autoreg.type_as(tgt_mask_padding); // [batch_size, 1, seq_len, seq_len]
            auto normalizer = (bos_tgt_ids != _pad_id).sum().item().toLong(); // #total - #mask
            Pack pack = {
                {"seq_ids", bos_tgt_ids},
                {"seq_mask", tgt_mask},
                {"normalizer", normalizer},
                {"labels", inp_ids}
            };
            return pack;
        }


        auto Trainer::step(data::Batch& batch, StatsCounter& stats, const Mode mode) -> Tensor {
            torch::AutoGradMode enable_grad(mode == Mode::TRAINING); // RAII
             if (_fp16_enabled) {  //__enter__()
                at::autocast::set_autocast_enabled(_device.type(), true);
            }
            if (mode == Mode::TRAINING){
                _optimizer->zero_grad();
            }
            batch.contiguous();
            batch = batch.to(_device);
            Pack pack;
            switch (_task_type) {
                case TaskType::LM:
                    pack = step_lm(batch, stats, mode);
                    break;
                case TaskType::NMT:
                    pack = step_nmt(batch, stats, mode);
                    break;
                default:
                    throw std::runtime_error("Unknown task type");
            }
            //////// DEBUGGING BEGIN ////////
            auto log_batch = [&](){
                // print batch for debugging
                for (auto i=0; i <batch.examples.size(); i++){
                    auto ex = batch.examples[i];
                    for (size_t j = 0; j < ex.fields.size(); ++j) {
                        std::cerr << "Field " << j << ": " << ex.fields[j] << std::endl;
                    }
                }
                for (const auto& [key, value] : pack) {
                    std::cerr << key << ": " << std::any_cast<Tensor>(value) << std::endl;
                }
            };
            bool debug_input = false;
            if (debug_input) {
                log_batch();
            }
            //////// DEBUGGING END////////


            auto result = _model->forward(pack);
            auto features = std::any_cast<Tensor>(result["result"]); // [batch_size, seq_len, model_dim]
            // skip the last token (EOS) in features, as it is not used in loss computation
            features = features.index({ Slice(), Slice(0, -1), Slice() }); // [batch_size, seq_len, model_dim]
            try {
                auto labels = std::any_cast<Tensor>(pack["labels"]);
                auto normalizer = std::any_cast<long>(pack["normalizer"]);
                auto loss = _loss_computer->compute(features, labels, normalizer, mode);
                stats.update(loss.item().toDouble(), labels.size(0), normalizer, _scheduler->get_last_rate());
                if (_fp16_enabled) {  // __exit__()
                    at::autocast::clear_cache();
                    at::autocast::set_autocast_enabled(_device.type(), false);
                }
                if (mode == Mode::TRAINING){
                    torch::nn::utils::clip_grad_norm_(_model->parameters(), _config["trainer"]["clip_grad"].as<f32>(5.0));
                    _optimizer->step();
                    _scheduler->step();
                }
                return loss;
            } catch (const BadBatchException& e) {
                log_batch();
                throw;
            }
        }


        void Trainer::train() {
            spdlog::info("Moving to device {}", _device == torch::kCUDA ? "cuda" : "cpu");
            _model->to(_device);
            size_t num_epochs = _config["trainer"]["epochs"].as<int>();
            auto log_frequency = _config["trainer"]["log_frequency"].as<std::string>("25u");
            auto checkpoint_frequency = _config["checkpoint"]["frequency"].as<size_t>(1000);
            auto validation_frequency = _config["validator"]["frequency"].as<size_t>(1000);
            spdlog::info("Training started; total epochs = {}; log_frequency={}", num_epochs, log_frequency);
            _model->train();

            Stopper stopper(_config["trainer"]["early_stopping"].as<int>(8));
            int16_t log_first = _config["trainer"]["log_first"].as<int16_t>(0);
            auto stats = StatsCounter(log_frequency, /*name=*/"Training", /*log_first=*/log_first);
            size_t n_skips = 0;
            size_t MAX_BAD_SKIPS = 5;
            for (i32 epoch = 0; epoch < num_epochs; epoch++) {
                for (auto batch : _data_loader.get_train_data()) {
                    try {
                        step(batch, stats, Mode::TRAINING);
                        n_skips = 0;
                    } catch (const BadBatchException& e) {
                        spdlog::error("Bad batch detected. Skipping batch. Error: {}", e.what());
                        n_skips++;
                        if (n_skips > MAX_BAD_SKIPS) {
                            spdlog::error("Too many bad batches. Aborting training training");
                            return;
                        }
                        continue;
                    }
                    batch.fields.clear(); // clear references to tensors

                    if (stats.step_num % validation_frequency == 0) {
                        bool is_early_stop = validate();
                        if (is_early_stop) {
                            spdlog::info("Early stopping at epoch {} step {}", epoch, stats.step_num);
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

        void Trainer::log_nmt_samples(){
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

        auto Trainer::validate(bool show_samples) -> bool {
            /**
             * Returns true if early stopping is triggered
            */
            torch::AutoGradMode enable_grad(false);
            _model->eval();
            if (show_samples) {
                switch (_task_type) {
                    case TaskType::NMT:
                        log_nmt_samples();
                        break;
                    case TaskType::LM:
                        spdlog::warn("Log samples is not supported for LM task");
                        break;
                    default:
                        spdlog::warn("Unknown task type");
                }
            }
            auto log_frequency = _config["validator"]["log_frequency"].as<std::string>("5000u");
            auto stats = StatsCounter(log_frequency, /*name*/"Validation");
            spdlog::info("Starting validation. log_frequency={}", stats.log_frequency);
            for (auto batch : _data_loader.get_validation_data()) {
                step(batch, stats, Mode::INFERENCE);
            }
            spdlog::info("Validation finished. Avg loss: {:.5f}", stats.avg_loss());
            f32 valid_loss = stats.avg_loss();
            // TODO: support multiple validation metrics
            // maybe we should save "last" by default.
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


}


// AMP/fp16 is discussed here https://discuss.pytorch.org/t/deploy-mixed-precision-model-in-libtorch/89046/5
