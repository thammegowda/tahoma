#pragma once

#include <stdexcept>
#include <torch/torch.h>

namespace tahoma::train {

    struct CriteriaContainer {
        nn::AnyModule train; // a single criterion for training
        std::map<std::string, nn::AnyModule> validation; // multiple criteria for validation, with names
    };

    class BadBatchException : public std::runtime_error {
    public:
        BadBatchException(const std::string& message) : std::runtime_error(message) {}
    };

    class LossComputer {
    public:
         LossComputer(nn::AnyModule& projector, std::shared_ptr<CriteriaContainer>& criteria, const i64 pad_id, const size_t chunk_size=0)
            : _projector{ projector }, _criteria{ criteria }, _pad_id{ pad_id }, _chunk_size{ chunk_size }
        {}

        torch::Tensor compute(torch::Tensor features, torch::Tensor labels, float normalizer = -1.0, Mode mode = Mode::TRAINING);

    private:
        nn::AnyModule _projector;
        std::shared_ptr<CriteriaContainer> _criteria;
        int64_t _pad_id;
        size_t _chunk_size;

        torch::Tensor compute_once(torch::Tensor features, torch::Tensor labels, float normalizer, Mode mode);
        torch::Tensor compute_chunked(torch::Tensor features, torch::Tensor labels, float normalizer, Mode mode);
    };

} // namespace tahoma::train
