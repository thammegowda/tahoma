#include <tuple>
#include <cmath>
#include <iostream>
#include <tuple>
#include <assert.h>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>

#include <tahoma.h>
#include <tahoma/layer/transformer.h>
#include <tahoma/model/transformer_lm.h>


using namespace tahoma;
using namespace torch::indexing;

namespace tahoma::model {

        TransformerLMImpl::TransformerLMImpl(const YAML::Node& args)
        : LanguageModel(args["model_dim"].as<size_t>(), args["vocab_size"].as<size_t>()),
            decoder {
                register_module("decoder", layer::transformer::TransformerEncoder(
                vocab_size,
                model_dim,
                args["attn_heads"].as<int>(),
                args["layers"].as<int>(),
                args["dropout"].as<double>()))
            }
        {}

        auto TransformerLMImpl::forward(Pack& args) -> Pack {
            auto seq_ids = std::any_cast<Tensor>(args["seq_ids"]); // [batch_size, seq_len]
            auto seq_mask = std::any_cast<Tensor>(args["seq_mask"]); // [batch_size, seq_len]
            // seq: [batch_size, seq_len]
            // return: [batch_size, tgt_len, tgt_vocab_size]
            auto output = decoder(seq_ids, seq_mask); // [batch_size, tgt_len, model_dim]
            //output = lm_head(output); // [batch_size, tgt_len, tgt_vocab_size]
            return { {"result", output} };
        }

} // namespace tahoma::model
