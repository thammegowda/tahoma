#pragma once


#include <sentencepiece_processor.h>
#include <rtg.hpp>

#include "../common/config.hpp"
#include "../common/data.hpp"
#include "./criterion.hpp"

using namespace rtg;


namespace rtg::train {

    
    class LossComputer {

    protected:
        nn::AnyModule projector;
        train::CrossEntropyLoss criterion;
        int64_t pad_id;
    public:
        LossComputer(nn::AnyModule& projector, train::CrossEntropyLoss& criterion, const i64 pad_id)
            : projector{ projector }, criterion{ criterion }, pad_id{ pad_id }
        {}

        virtual auto compute(Tensor features, Tensor labels, f32 normalizer = -1, Mode mode = Mode::TRAINING) -> Tensor = 0;
    };


    class SimpleLossComputer : public LossComputer {

    public:
        SimpleLossComputer(nn::AnyModule& projector, train::CrossEntropyLoss& criterion, i64 pad_id)
            : LossComputer{ projector, criterion, pad_id }
        {}

        auto compute(Tensor features, Tensor labels, f32 normalizer = -1.0, Mode mode = Mode::TRAINING) -> Tensor override {
            auto output = projector.forward(features);
            auto output_flat = output.view({ output.size(0) * output.size(1), -1 }); // [batch_size * seq_len, vocab_size]
            auto labels_flat = labels.reshape({ -1 }); // [batch_size * seq_len]
            if (normalizer <= 0.0) { // self computed normalizer
                normalizer = (labels_flat != pad_id).sum().item().toInt(); // #total - #mask
            }
            Tensor loss = criterion(output_flat, labels_flat, normalizer);  // [batch_size * seq_len]
            if (mode == Mode::TRAINING) {
                loss.backward();
            }
            return loss;
        }
    };


    class ChunkedLossComputer : public SimpleLossComputer {

    protected:
        size_t chunk_size;

    public:
        ChunkedLossComputer(nn::AnyModule& projector, train::CrossEntropyLoss& criterion, int64_t pad_id, size_t chunk_size)
            : SimpleLossComputer{ projector, criterion, pad_id }, chunk_size{ chunk_size }
        {
            if (chunk_size <= 0) {
                throw runtime_error("chunk_size must be > 0");
            }
        }

        auto compute(Tensor features, Tensor labels, f32 normalizer = -1.0, Mode mode = Mode::TRAINING) -> Tensor override {
            /**
             * Compute loss in chunks to avoid OOM
             * features: [batch_size, seq_len, hidden_size]
             * labels: [batch_size, seq_len]
             * normalizer: total number of tokens in batch. If not provided, it is computed based on pad_id
            */
            const size_t seq_len = features.size(1);
            const auto total_chunks = ceil((1.0 * seq_len) / chunk_size);
            Tensor total_loss = torch::tensor(0.0, torch::device(features.device()).dtype(torch::kFloat32));
            total_loss.requires_grad_(false); // cant do backward on this loss value

            // disconnect graph, ulate grad across chunks, and then do backward
            Tensor features_isolated = features.detach().clone();
            features_isolated.requires_grad_(true);
            for (auto chunk_idx = 0z; chunk_idx < total_chunks; chunk_idx++) {
                auto start = chunk_idx * chunk_size;
                auto end = min((chunk_idx + 1) * chunk_size, seq_len);
                auto chunk_features = features_isolated.index({ Slice(), Slice(start, end), Slice() });
                auto chunk_labels = labels.index({ Slice(), Slice(start, end) });
                auto chunk_loss = SimpleLossComputer::compute(chunk_features, chunk_labels, normalizer, mode);
                total_loss += chunk_loss.item().toFloat();
            }

            if (mode == Mode::TRAINING) {
                features.backward(features_isolated.grad().data());
            }
            return total_loss;
        }
    };

}