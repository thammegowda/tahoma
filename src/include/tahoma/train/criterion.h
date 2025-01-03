#pragma once
#include <tahoma.h>


using namespace tahoma;
namespace F = torch::nn::functional;

namespace tahoma::train {

    namespace optim = torch::optim;
    enum class DataType {
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
        virtual auto forward(Tensor input, Tensor target, f32 normalizer, std::optional<torch::Tensor> mask = std::nullopt) -> Tensor = 0;
    };

    class CrossEntropyLossImpl : public CriterionImpl {
    protected:
        f32 _label_smooth_rate;
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
        auto forward(Tensor input, Tensor target, f32 normalizer, std::optional<Tensor> mask) -> Tensor override;
    };
    TORCH_MODULE(CrossEntropyLoss);

    class KLDivergenceImpl : public CriterionImpl {
    protected:
        i64 _num_labels;
        f32 _label_smooth_rate;
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
        auto forward(Tensor input, Tensor target, f32 normalizer, std::optional<Tensor> mask) -> Tensor override;
    };
    TORCH_MODULE(KLDivergence);

    class LRScheduler {
    protected:
        optim::Optimizer& _optimizer;
        i64 _current_step;
        f32 _last_rate;

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
        i32 warmup_steps;
        f32 init_lr;
        f32 peak_lr;
        f32 constant;

        InverseSqrtSchedulerOptions() = default;
        InverseSqrtSchedulerOptions(const YAML::Node& config);
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

        auto get_rate() -> f32 override;
    };

    struct NoamOptions {
        i32 warmup_steps;
        i32 model_dim;
        f32 constant;

        NoamOptions() = default;
        NoamOptions(const YAML::Node& config);
        void validate();
    };

    class NoamScheduler : public LRScheduler {
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

        auto get_rate() -> f32 override;
    };

}