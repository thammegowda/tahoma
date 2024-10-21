#pragma once
#include <tahoma.h>
#include <tahoma/model.h>
#include <tahoma/layer/transformer.h>

using namespace tahoma;
namespace tahoma::model {
    struct TransformerNMTImpl: public LanguageModel {
        size_t src_vocab_size;
        size_t tgt_vocab_size;
        size_t model_dim;

        layer::transformer::TransformerEncoder encoder;
        layer::transformer::TransformerDecoder decoder;

        TransformerNMTImpl(const YAML::Node& args);

        auto task_type() -> TaskType override {
            return TaskType::NMT;
        }

        virtual auto forward(Pack& args) -> Pack override;
    };
    TORCH_MODULE(TransformerNMT);
}