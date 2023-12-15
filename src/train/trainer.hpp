#pragma once
#include <iostream>
#include <coroutine>
#include <ranges>
#include <vector>
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
#include "../model/transformer_nmt.hpp"
#include "./utils.hpp"
#include "./stats_counter.hpp"

namespace nn = torch::nn;
namespace optim = torch::optim;
namespace sp = sentencepiece;
namespace fs = std::filesystem;

using namespace std;
using namespace torch::indexing;
using namespace rtg;
using namespace chrono;

namespace rtg::train {


    class Decoder {
    private:
        rtg::model::TransformerNMT _model;
        nn::AnyModule _lm_head;
        vector<shared_ptr<sp::SentencePieceProcessor>> _vocabs;
        torch::Device _device;
    public:
        Decoder(rtg::model::TransformerNMT _model, nn::AnyModule lm_head, vector<shared_ptr<sp::SentencePieceProcessor>> _vocabs, torch::Device _device):
            _model {_model}, _lm_head {lm_head}, _vocabs{_vocabs}, _device{_device}
        {
            if (_vocabs.size() != 2){
                throw std::invalid_argument("Vocab size must be 2, but found " + _vocabs.size());
            }
        }

        auto greedy_decode(str src, i32 max_len=128) -> str{
            auto src_vocab = _vocabs[0];
            auto tgt_vocab = _vocabs[1];
            vector<int> src_ids_vec = _vocabs[0]->EncodeAsIds(src);
            auto src_ids = torch::tensor(src_ids_vec, torch::dtype(torch::kInt64).device(_device)).unsqueeze(0); // [1, src_len]

            auto src_mask = (src_ids == src_vocab->pad_id()).unsqueeze(0).unsqueeze(1); // [batch=1, 1, 1, src_len]  
            auto memory = _model ->encoder(src_ids, src_mask);
            std::cerr << "src: " << utils::tensor_shape(src_ids) << "; mask:" << utils::tensor_shape(src_mask) 
                << "; memory: " << utils::tensor_shape(memory) << std::endl;
            
            auto tgt_ids = torch::full({src_ids.size(0), 1}, tgt_vocab->bos_id(), torch::dtype(torch::kInt64).device(_device));
            for (int i=0; i < max_len; i++){
                auto tgt_mask = subsequent_mask(tgt_ids.size(1), _device).unsqueeze(0).unsqueeze(1);  // [batch=1, head=1, tgt_len, tgt_len]
                std::cerr << "src_mem: " << utils::tensor_shape(memory) << "; src_mask:" << utils::tensor_shape(src_mask) 
                << "; tgt_ids: " << utils::tensor_shape(tgt_ids) << "; tgt_mask: " << utils::tensor_shape(tgt_mask) << std::endl;
                auto features = _model->decoder(tgt_ids, memory, tgt_mask, src_mask);
                features = features.index({Slice(), -1, Slice()});
                auto output = _lm_head.forward(features);
                auto next_token = output.argmax(-1);
                std::cerr << "next_token: " << next_token << std::endl;

                // TODO max and compute score
                tgt_ids = torch::cat({tgt_ids, next_token}, 1);
                // TODO: Halt on EOS
            }
            std::vector<int> tgt_ids_vec(tgt_ids.data_ptr<i64>(), tgt_ids.data_ptr<i64>() + tgt_ids.numel());
            auto tgt_tokens = _vocabs[1]->DecodeIds(tgt_ids_vec);
            return tgt_tokens;
        }
    };

    template <typename M>
    class Trainer {
    protected:
        fs::path _work_dir;
        config::Config _config;
        M _model;
        nn::AnyModule _projector;
        shared_ptr<optim::Optimizer> _optimizer;
        shared_ptr<train::LRScheduler> _scheduler;
        rtg::data::DataLoader _data_loader;

        bool _fp16_enabled;
        int64_t _pad_id = 0;
        int64_t _bos_id = 1;
        shared_ptr<LossComputer> _loss_computer;
        torch::Device _device{ torch::cuda::is_available() ? "cuda" : "cpu" };
        data::Batch _sample_batch;

    public:
        Trainer(fs::path work_dir, config::Config conf)
            : _work_dir{ work_dir },
            _config{ conf },
            _model{ init_model<M>(_config, _device) },
            _projector{ _model->lm_head },
            _optimizer{ init_optimizer(_config, _model) },
            _scheduler{ init_scheduler(_config, *_optimizer) },
            _data_loader{ rtg::data::DataLoader(_config) },
            _fp16_enabled{ _config["trainer"]["fp16"].as<bool>(false) },
            _pad_id {_data_loader.vocabs[1]->pad_id()},   //vocabs[1] is the target vocab
            _bos_id {_data_loader.vocabs[1]->bos_id()},
            _loss_computer{  init_loss_computer(_config, _projector, _pad_id) },

            _sample_batch{ _data_loader.get_samples(_config["validator"]["data"].as<vector<string>>(), 5) }
        {
            spdlog::info("Trainer initialized; work_dir={} device={}; fp16={}",
                work_dir, _device == torch::kCUDA ? "cuda" : "cpu", _fp16_enabled);
            if (torch::cuda::is_available()) {
                vector<string> device_ids;
                for (auto i=0; i < torch::cuda::device_count(); i++){
                    device_ids.push_back(fmt::format("{} ", i));
                }
                spdlog::info("CUDA devices: {}", fmt::join(device_ids, ", "));
            }
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
            auto checkpoint_file = _work_dir / filename;
            spdlog::info("Saving checkpoint to {}", checkpoint_file);
            torch::save(_model, checkpoint_file.string());
        }

        auto step(data::Batch& batch, StatsCounter& stats, const Mode mode = Mode::TRAINING) -> Tensor {
             if (_fp16_enabled) {  //__enter__()
                at::autocast::set_enabled(true);
            }
            batch.contiguous();
            batch = batch.to(_device);
            auto src_ids = batch.fields[0];  // [batch_size, seq_len]
            auto tgt_ids = batch.fields[1];   // [batch_size, seq_len]
            //FIME: add bos and eos tokens to tgt_ids. Offset tgt_ids by 1
            auto _bos_col = torch::full({ tgt_ids.size(0), 1 }, _bos_id, torch::dtype(torch::kInt64).device(_device));
            tgt_ids = torch::cat({_bos_col, tgt_ids}, 1);
            
            auto src_mask = (src_ids == _pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len]
            auto tgt_mask_padding = (tgt_ids == _pad_id).unsqueeze(1).unsqueeze(2); // [batch_size, 1, 1, seq_len] 
            auto tgt_mask_autoreg = subsequent_mask(tgt_ids.size(1), _device).unsqueeze(0).unsqueeze(1); // [1, 1, seq_len, seq_len] 
            auto tgt_mask = tgt_mask_padding | tgt_mask_autoreg.type_as(tgt_mask_padding); // [batch_size, 1, seq_len, seq_len]
            auto normalizer = (tgt_ids != _pad_id).sum().item().toInt(); // #total - #mask
            auto output = _model(src_ids, tgt_ids, src_mask, tgt_mask);
            auto loss = _loss_computer->compute(output, tgt_ids, normalizer, mode);

            if (_fp16_enabled) {  // __exit__()
                at::autocast::clear_cache();
                at::autocast::set_enabled(false);
            }
            if (mode == Mode::TRAINING){
                _optimizer->step();
                _scheduler->step();
                _optimizer->zero_grad();
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
                        float valid_loss = validate();
                        // TODO: support multiple validation metrics
                        switch (stopper.is_stop(valid_loss)) {
                            case StopperStatus::NEW_BEST: // new best loss
                                save_checkpoint("best");
                                break;
                            case StopperStatus::STOP: // stop training
                                save_checkpoint("last");
                                LOG::info("Early stopping at epoch {} step {}", epoch, stats.step_num);
                                return;
                            // else continue training
                        }
                        _model->train();
                    }
                    if (stats.step_num % checkpoint_frequency == 0) {
                        save_checkpoint(fmt::format("step.{}", stats.step_num));
                    }
                }
            }
        }

        void log_samples(){
            auto decoder = Decoder(_model, _projector, _data_loader.vocabs, _device);
           for (auto i=0; i < _sample_batch.size(); i++) {
                auto ex = _sample_batch.examples[i];
                str src = ex.fields[0];
                str ref = ex.fields[1];
                spdlog::info("[{}] SRC: {}", i, src);
                spdlog::info("[{}] REF: {}", i, ref);
                str hyp = decoder.greedy_decode(src);
                spdlog::info("[{}] HYP: {}", i, hyp);
           }
        }

        auto validate(bool show_samples=false) -> float {
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
            return stats.avg_loss();
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
    auto trainer = train::Trainer<model::TransformerNMT>(work_dir, config_file);
    trainer.train();
    spdlog::info("main finished..");
    return 0;
}


// AMP/fp16 is discussed here https://discuss.pytorch.org/t/deploy-mixed-precision-model-in-libtorch/89046/5 
