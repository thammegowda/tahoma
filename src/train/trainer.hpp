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

    template <typename M, typename C>
    class Trainer {
    protected:
        fs::path work_dir;
        config::Config config;
        M model;
        C criterion;
        nn::AnyModule projector;
        shared_ptr<optim::Optimizer> optimizer;
        shared_ptr<optim::LRScheduler> scheduler;
        rtg::data::DataLoader data_loader;

        bool fp16_enabled;
        int64_t pad_id = 0;
        int64_t bos_id = 1;
        shared_ptr<LossComputer> loss_computer;
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
            fp16_enabled{ config["trainer"]["fp16"].as<bool>(false) },
            pad_id {data_loader.vocabs[1]->pad_id()},   //vocabs[1] is the target vocab
            bos_id {data_loader.vocabs[1]->bos_id()},
            loss_computer{  init_loss_computer(config, projector, criterion, pad_id) }
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
            auto loss = loss_computer->compute(output, tgt_ids, normalizer, mode);
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
                        { // scope for AutoGradMode
                            torch::AutoGradMode enable_grad(false);
                            model->eval();
                            auto valid_loss = validate();
                            switch (stopper.is_stop(valid_loss)) {
                                case StopperStatus::NEW_BEST: // new best loss
                                    save_checkpoint("best");
                                    break;
                                case StopperStatus::STOP: // stop training
                                    LOG::info("Early stopping at epoch {} step {}", epoch, stats.step_num);
                                    return;
                               // else continue training
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
