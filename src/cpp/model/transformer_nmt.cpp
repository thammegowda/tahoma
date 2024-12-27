#include <tuple>
#include <cmath>
#include <iostream>
#include <tuple>
#include <assert.h>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>

#include <tahoma.h>
#include <tahoma/layer/transformer.h>
#include <tahoma/model/transformer_nmt.h>


namespace nn = torch::nn;
namespace F = torch::nn::functional;
using namespace tahoma;
using namespace torch::indexing;
using namespace tahoma::layer;

namespace tahoma::model {

    // ctor
    TransformerNMTImpl::TransformerNMTImpl(const YAML::Node& args):
        LanguageModel(args["model_dim"].as<size_t>(), args["tgt_vocab_size"].as<size_t>()),
        src_vocab_size { args["src_vocab_size"].as<size_t>() },
        tgt_vocab_size { args["tgt_vocab_size"].as<size_t>() },
        model_dim { args["model_dim"].as<size_t>() },

        encoder {
            register_module("encoder", layer::transformer::TransformerEncoder(
            src_vocab_size,
            model_dim,
            model_dim * 4,
            args["attn_heads"].as<int>(),
            args["encoder_layers"].as<int>(),
            args["dropout"].as<double>()))
        },
        decoder {
            register_module("decoder", layer::transformer::TransformerDecoder(
            tgt_vocab_size,
            model_dim,
            model_dim * 4,
            args["attn_heads"].as<int>(),
            args["decoder_layers"].as<int>(),
            args["dropout"].as<double>()))
        }
    {}


    auto TransformerNMTImpl::forward(Pack& args) -> Pack {
        auto src = std::any_cast<Tensor>(args["src"]); // [batch_size, src_len]
        auto tgt = std::any_cast<Tensor>(args["tgt"]); // [batch_size, tgt_len]
        auto src_mask = std::any_cast<Tensor>(args["src_mask"]); // [batch_size, src_len]
        auto tgt_mask = std::any_cast<Tensor>(args["tgt_mask"]); // [batch_size, tgt_len]
        auto memory = encoder(src, src_mask); // [batch_size, src_len, model_dim]
        auto output = decoder(tgt, tgt_mask, memory, src_mask); // [batch_size, tgt_len, model_dim]
        //output = lm_head(output); // [batch_size, tgt_len, tgt_vocab_size]
        return { {"result", output} };
    }

}
