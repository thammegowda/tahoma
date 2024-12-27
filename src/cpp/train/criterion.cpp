#include <optional>
#include <algorithm>
#include <cmath>
#include <torch/torch.h>
#include <tahoma.h>
#include <tahoma/train/criterion.h>


namespace nn = torch::nn;
namespace optim = torch::optim;
namespace F = torch::nn::functional;
using namespace torch::indexing;


namespace tahoma::train {

    auto CrossEntropyLossImpl::forward(Tensor input, Tensor target, f32 normalizer, std::optional<Tensor> mask) -> Tensor {
        // input: [batch_size, seq_len, vocab_size]
        // target: [batch_size, seq_len]
        auto num_labels = input.size(-1);
        auto input_flat = input.view({ -1, num_labels }); // [batch_size * seq_len, vocab_size]
        auto target_flat = target.reshape({ -1 }); // [batch_size * seq_len]
        Tensor loss = F::cross_entropy(input_flat, target_flat, _options);  // [batch_size * seq_len]
        if (mask) {
            loss.masked_fill_(mask.value().reshape({ -1 }), 0.0);
        }
        else if (_ignore_index >= 0) { // auto deduce mask from target
            loss.masked_fill_(target_flat == _ignore_index, 0.0);
        }
        loss = loss.sum() / normalizer;
        return loss;
    }



    auto KLDivergenceImpl::forward(Tensor input, Tensor target, f32 normalizer, std::optional<Tensor> mask) -> Tensor {
        // input: [batch_size, seq_len, vocab_size]
        // target: [batch_size, seq_len]

        auto input_flat = input.view({ -1, _num_labels }); // [batch_size * seq_len, vocab_size]
        auto target_flat_1hot = target.reshape({ -1, 1 }); // [batch_size * seq_len, 1]
        // flat => batch_size * seq_len dims flattened
        // 1hot => one hot encoding, in sparse mode, just indices are stored
        // smooth => dense distribution obtained by label smoothing, all values are equal to label_smooth_rate / (num_labels - num_exclusions)

        // smooth the target distribution
        auto target_flat_smooth = torch::full_like(input_flat, _label_smooth_rate / (_num_labels - _num_exclusions));
        target_flat_smooth.scatter_(1, target_flat_1hot, 1.0 - _label_smooth_rate);
        if (_ignore_index >= 0) { // exclude padding token placeholder (column) in target distribution
            target_flat_smooth.index_put_({ Slice(), _ignore_index }, 0.0);
            input_flat.index_put_({ Slice(), _ignore_index }, 0.0);   // caution: this is inplace operation on inputs!
        }
        auto loss = F::kl_div(input_flat.log_softmax(-1), target_flat_smooth, _options);
        if (mask) {
            loss.masked_fill_(mask.value().reshape({ -1 }), 0.0);
        }
        else if (_ignore_index >= 0) { //exclude padding tokens (rows) in input
            loss.masked_fill_(target_flat_1hot == _ignore_index, 0.0);
        }
        loss = loss.sum() / normalizer;
        return loss;
    }


    InverseSqrtSchedulerOptions::InverseSqrtSchedulerOptions(const YAML::Node& config){
        warmup_steps = config["warmup_steps"].as<i32>(0);
        init_lr = config["init_lr"].as<f32>(0.0);
        peak_lr = config["peak_lr"].as<f32>(0.0);
        constant = config["constant"].as<f32>(1.0);
        if (warmup_steps <= 0.0 || init_lr < 0.0 || peak_lr <= 0.0 || constant <= 0.0) {
            throw std::invalid_argument(fmt::format(
                "All values must be positive; warmup_steps: {}, init_lr: {}, peak_lr: {}, constant: {}",
                warmup_steps, init_lr, peak_lr, constant));
        }
    }

    auto InverseSqrtScheduler::get_rate() -> f32 {
        f32 rate;
        if (_current_step < _options.warmup_steps) {
            rate = _options.init_lr + (_options.peak_lr - _options.init_lr) * _current_step / _options.warmup_steps;
        }
        else {
            rate = _options.peak_lr * pow(_current_step, -0.5);
        }
        return rate * _options.constant;
    }


    NoamOptions::NoamOptions(const YAML::Node& config) {
        warmup_steps = config["warmup_steps"].as<i32>(0);
        model_dim = config["model_dim"].as<i32>(0);
        constant = config["constant"].as<f32>(1.0);
        if (warmup_steps <= 0 || model_dim <= 0 || constant <= 0.0) {
            throw std::invalid_argument(fmt::format(
                "All values must be positive; warmup_steps: {}, model_dim: {}, constant: {}",
                warmup_steps, model_dim, constant));
        }
    }


    auto NoamScheduler::get_rate() -> f32 {
        //  return self.constant * self.model_dim**-0.5 * min(step**-0.5, step * self.warmup**-1.5)
        f32 rate = _options.constant
            * std::pow(_options.model_dim, -0.5)
            * std::min( pow(_current_step, -0.5),
                _current_step * pow(_options.warmup_steps, -1.5)
            );
        return rate;
    }

}