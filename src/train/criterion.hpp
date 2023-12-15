#pragma once

#include <optional>
#include <rtg.hpp>
#include <torch/torch.h>


namespace nn = torch::nn;
namespace optim = torch::optim;
namespace F = torch::nn::functional;
using namespace torch::indexing;


namespace rtg::train {

    enum DataType {
        LOGITS,
        SOFTMAX,
        LOG_SOFTMAX,
        SIGMOID,
    };

    class CriterionImpl : public nn::Module {
    protected:
        str _name;
        i64 _ignore_index;
        DataType _input_type;

    public:
        CriterionImpl(str name, i64 ignore_index, DataType input_type = DataType::LOGITS) :
            _name{ name },
            _ignore_index{ ignore_index },
            _input_type{ input_type }
        {}
        virtual auto forward(Tensor input, Tensor target, f32 normalizer, optional<Tensor> mask = nullopt) -> Tensor = 0;
    };
    //TORCH_MODULE(Criterion);


    class CrossEntropyLossImpl : public CriterionImpl {
    protected:
        f32 _label_smooth_rate = 0.0;
        F::CrossEntropyFuncOptions _options;

    public:
        CrossEntropyLossImpl(i64 ignore_index = -100, f32 label_smooth_rate = 0.0) :
            CriterionImpl("cross_entropy", ignore_index, DataType::LOGITS),
            _label_smooth_rate{ label_smooth_rate },
            _options{ F::CrossEntropyFuncOptions()
                        .reduction(torch::kNone)
                        .label_smoothing(label_smooth_rate)
                        .ignore_index(ignore_index)
            }
        {}

        auto forward(Tensor input, Tensor target, f32 normalizer, optional<Tensor> mask) -> Tensor {
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
    };
    TORCH_MODULE(CrossEntropyLoss);

    class KLDivergenceImpl : public CriterionImpl {
    protected:
        f32 _label_smooth_rate = 0.0;
        i64 _num_labels;
        i64 _num_exclusions;
        F::KLDivFuncOptions _options;

    public:
        KLDivergenceImpl(i64 num_labels, i64 ignore_index = -100, f32 label_smooth_rate = 0.0) :
            CriterionImpl("kl_divergence", ignore_index, DataType::LOGITS),
            _num_labels{ num_labels },
            _label_smooth_rate{ label_smooth_rate },
            _num_exclusions{ _ignore_index >= 0 ? 2 : 1 },  // the hot label and ignore_index (optional)
            _options{ F::KLDivFuncOptions().reduction(torch::kNone) }
        {
            if (_label_smooth_rate <= 0.0) {
                throw std::invalid_argument("label_smooth_rate must be positive.");
            }
        }

        auto forward(Tensor input, Tensor target, f32 normalizer, optional<Tensor> mask) -> Tensor {
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
    };
    TORCH_MODULE(KLDivergence);


    class LRScheduler {
    protected:
        i64 _current_step;
        optim::Optimizer& _optimizer;
        f32 _last_rate = 0.0;

    public:
        LRScheduler(optim::Optimizer& optimizer, i64 current_step = 0) :
            _optimizer{ optimizer },
            _current_step{ current_step }
        {}

        virtual auto get_rate() -> f32 = 0;

        void step() {
            _current_step++;
            _last_rate = get_rate(); // currently setting the same rates for all
            for (auto& param_group : _optimizer.param_groups()) {
                param_group.options().set_lr(_last_rate);
            }
        }

        auto get_last_rate() -> f32 {
            return _last_rate;
        }

    };


    struct InverseSqrtSchedulerOptions {
        i32 warmup_steps = 1000;
        f32 init_lr = 1e-5;
        f32 peak_lr = 1e-3;
        f32 constant = 1.0;

        InverseSqrtSchedulerOptions() {}

        InverseSqrtSchedulerOptions(const YAML::Node& config){
            warmup_steps = config["warmup_steps"].as<i32>(0);
            init_lr = config["init_lr"].as<f32>(0.0);
            peak_lr = config["peak_lr"].as<f32>(0.0);
            constant = config["constant"].as<f32>(1.0);
            validate();
        }

        void validate() {
            if (warmup_steps <= 0.0 || init_lr < 0.0 || peak_lr <= 0.0 || constant <= 0.0) {
                throw std::invalid_argument(fmt::format(
                    "All values must be positive; warmup_steps: {}, init_lr: {}, peak_lr: {}, constant: {}",
                    warmup_steps, init_lr, peak_lr, constant));
            }
        }
    };

    class InverseSqrtScheduler : public LRScheduler {
    protected:
        InverseSqrtSchedulerOptions _options;
    public:

        InverseSqrtScheduler(optim::Optimizer& optimizer, i64 current_step, InverseSqrtSchedulerOptions options) :
            LRScheduler(optimizer, current_step),
            _options{ options }
        {}

        // TODO: implcit conversion from YAML::Node to Options 
        InverseSqrtScheduler(optim::Optimizer& optimizer, i64 current_step, const YAML::Node& config) :
            InverseSqrtScheduler(optimizer, current_step, InverseSqrtSchedulerOptions(config))
        {}

        auto get_rate() -> f32 {
            f32 rate;
            if (_current_step < _options.warmup_steps) {
                rate = _options.init_lr + (_options.peak_lr - _options.init_lr) * _current_step / _options.warmup_steps;
            }
            else {
                rate = _options.peak_lr * pow(_current_step, -0.5);
            }
            return rate * _options.constant;
        }
    };


    struct NoamOptions {
        i32 warmup_steps = 1'000;
        i32 model_dim = 512;
        f32 constant = 1.0;

        NoamOptions() {}

        NoamOptions(const YAML::Node& config) {
            warmup_steps = config["warmup_steps"].as<i32>(0);
            model_dim = config["model_dim"].as<i32>(0);
            constant = config["constant"].as<f32>(1.0);
            validate();
        }

        void validate() {
            if (warmup_steps <= 0 || model_dim <= 0 || constant <= 0.0) {
                throw std::invalid_argument(fmt::format(
                    "All values must be positive; warmup_steps: {}, model_dim: {}, constant: {}",
                    warmup_steps, model_dim, constant));
            }
        }
    };

    class NoamScheduler: public LRScheduler {
    protected:
        NoamOptions _options;
    public:
        NoamScheduler(optim::Optimizer& optimizer, i64 current_step, NoamOptions options) :
            LRScheduler(optimizer, current_step),
            _options{ options }
        {}
         NoamScheduler(optim::Optimizer& optimizer, i64 current_step, const YAML::Node& config):
            NoamScheduler(optimizer, current_step, NoamOptions(config))
        {}

        auto get_rate() -> f32 {
            //  return self.constant * self.model_dim**-0.5 * min(step**-0.5, step * self.warmup**-1.5)
            f32 rate = _options.constant 
                * pow(_options.model_dim, -0.5) 
                * min( pow(_current_step, -0.5), 
                    _current_step * pow(_options.warmup_steps, -1.5)
                );
            return rate;
        }
    };


}