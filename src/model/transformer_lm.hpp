#pragma once
#include <tuple>
#include <cmath>
#include <iostream>
#include <tuple>
#include <assert.h>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include "../common/config.hpp"
#include "../layer/transformer.hpp"


namespace nn = torch::nn;
namespace F = torch::nn::functional;
using namespace tahoma;
using namespace torch::indexing;
using namespace tahoma::layer;

namespace tahoma::model {


    struct IModel: public nn::Module {
        IModel() = default;
        ~IModel() = default;
        virtual auto task_type() -> TaskType = 0;
        // virtual functions and templates dont mix. so we use std::any for the return type
        virtual auto forward(Pack& args) -> Pack = 0;
    };


    struct LanguageModel: public IModel {
        size_t vocab_size;
        size_t model_dim;
        nn::Linear lm_head;

        LanguageModel(size_t model_dim, size_t vocab_size):
            model_dim { model_dim },
            vocab_size { vocab_size },
            lm_head { register_module("lm_head", nn::Linear(nn::LinearOptions(model_dim, vocab_size))) }
        {}
    };

    struct TransformerLMImpl: public LanguageModel {

        TransformerEncoder decoder;   // encoder is decoder now since it doesnt have cross-attention (src_attn)

        TransformerLMImpl(const YAML::Node& args):
            LanguageModel(args["model_dim"].as<size_t>(), args["vocab_size"].as<size_t>()),
            decoder {
                register_module("decoder", TransformerEncoder(
                vocab_size,
                model_dim,
                args["attn_heads"].as<int>(),
                args["layers"].as<int>(),
                args["dropout"].as<double>()))
            }
        {}

        // task_type
        auto task_type() -> TaskType override {
            return TaskType::LM;
        }

        virtual auto forward(Pack& args) -> Pack override {
            auto seq = std::any_cast<Tensor>(args["seq"]); // [batch_size, seq_len]
            auto seq_mask = std::any_cast<Tensor>(args["seq_mask"]); // [batch_size, seq_len]
            // seq: [batch_size, seq_len]
            // return: [batch_size, tgt_len, tgt_vocab_size]
            auto output = decoder(seq, seq_mask); // [batch_size, tgt_len, model_dim]
            //output = lm_head(output); // [batch_size, tgt_len, tgt_vocab_size]
            return { {"result", output} };
        }
    };
    TORCH_MODULE(TransformerLM);

}
