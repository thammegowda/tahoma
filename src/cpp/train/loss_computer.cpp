#include <vector>
#include <map>
#include <memory>
#include <optional>

#include <sentencepiece_processor.h>
#include <tahoma.h>
#include <tahoma/train/criterion.h>
#include <tahoma/train/loss_computer.h>

using namespace tahoma;


namespace tahoma::train {

    auto LossComputer::compute_once(Tensor features, Tensor labels, f32 normalizer, Mode mode) -> Tensor {
        auto output = _projector.forward(features);
        auto output_flat = output.view({ output.size(0) * output.size(1), -1 }); // [batch_size * seq_len, vocab_size]
        auto labels_flat = labels.reshape({ -1 }); // [batch_size * seq_len]
        if (normalizer <= 0.0) { // self computed normalizer
            normalizer = (labels_flat != _pad_id).sum().item().toInt(); // #total - #mask
        }
        // get first criterion for validation
        auto& criterion = (mode == Mode::TRAINING) ? _criteria->train : _criteria->validation.begin()->second;
        std::optional<Tensor> mask;
        Tensor loss = criterion.forward(output_flat, labels_flat, normalizer, mask);  // [batch_size * seq_len]
        if (mode == Mode::TRAINING) {
            if (torch::isfinite(loss).item().toBool()){
                loss.backward();
            } else {
                throw BadBatchException("Loss is not finite");
            }
        }
        return loss;
    }

    auto LossComputer::compute_chunked(Tensor features, Tensor labels, f32 normalizer, Mode mode) -> Tensor {
        /**
         * Compute loss in chunks to avoid OOM
         * features: [batch_size, seq_len, hidden_size]
         * labels: [batch_size, seq_len]
         * normalizer: total number of tokens in batch. If not provided, it is computed based on pad_id
        */
        if (_chunk_size < 0) {
            throw std::invalid_argument("chunk_size must be >= 0");
        }
        const size_t seq_len = features.size(1);
        const auto total_chunks = ceil((1.0 * seq_len) / _chunk_size);
        Tensor total_loss = torch::tensor(0.0, torch::device(features.device()).dtype(torch::kFloat32));
        total_loss.requires_grad_(false); // cant do backward on this loss value

        // disconnect graph, ulate grad across chunks, and then do backward
        Tensor features_isolated = features.detach().clone();
        features_isolated.requires_grad_(true);
        for (auto chunk_idx = 0z; chunk_idx < total_chunks; chunk_idx++) {
            auto start = chunk_idx * _chunk_size;
            auto end = std::min((chunk_idx + 1) * _chunk_size, seq_len);
            auto chunk_features = features_isolated.index({ Slice(), Slice(start, end), Slice() });
            auto chunk_labels = labels.index({ Slice(), Slice(start, end) });
            auto chunk_loss = compute_once(chunk_features, chunk_labels, normalizer, mode);
            if (mode == Mode::TRAINING && !torch::isfinite(chunk_loss).item().toBool()){
                throw BadBatchException("Loss is not finite");
            }
            total_loss += chunk_loss.item().toFloat();
        }

        if (mode == Mode::TRAINING) {
            features.backward(features_isolated.grad().data());
        }
        return total_loss;
    }


    auto LossComputer::compute(Tensor features, Tensor labels, f32 normalizer, Mode mode) -> Tensor {
        if (_chunk_size <= 0) {
            return compute_once(features, labels, normalizer, mode);
        } else {
            return compute_chunked(features, labels, normalizer, mode);
        }
    }
}

