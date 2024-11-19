
#pragma once
#include <torch/torch.h>
#include <tahoma.h>
#include <tahoma/model.h>
#include <tahoma/layer/transformer.h>

using namespace tahoma;

namespace tahoma::model {
    struct TransformerLMImpl : public LanguageModel {
        layer::transformer::TransformerEncoder decoder;

        TransformerLMImpl(const YAML::Node& args);

        auto task_type() -> TaskType override {
            return TaskType::LM;
        }

        auto forward(Pack& args) -> Pack override;
    };
    TORCH_MODULE(TransformerLM);

}