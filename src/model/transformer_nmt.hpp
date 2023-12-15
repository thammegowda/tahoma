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
using namespace rtg;
using namespace torch::indexing;
using namespace rtg::layer;

namespace rtg::model {

    struct TransformerNMTImpl : public nn::Module {
        int src_vocab_size;
        int tgt_vocab_size;
        int model_dim;

        TransformerEncoder encoder;
        TransformerDecoder decoder;
        nn::Linear lm_head;

        TransformerNMTImpl(const YAML::Node& args):
            src_vocab_size { args["src_vocab_size"].as<int>() },
            tgt_vocab_size { args["tgt_vocab_size"].as<int>() },
            model_dim { args["model_dim"].as<int>() },

            encoder {
                register_module("encoder", TransformerEncoder(
                src_vocab_size,
                model_dim,
                args["attn_head"].as<int>(),
                args["encoder_layers"].as<int>(),
                args["dropout"].as<double>()))
            },
            decoder {
                register_module("decoder",  TransformerDecoder(
                tgt_vocab_size,
                model_dim,
                args["attn_head"].as<int>(),
                args["decoder_layers"].as<int>(),
                args["dropout"].as<double>()))
            },
            lm_head { register_module("lm_head", nn::Linear(nn::LinearOptions(model_dim, tgt_vocab_size))) }
        {}

        auto forward(torch::Tensor& src, torch::Tensor& src_mask,
                    torch::Tensor& tgt, torch::Tensor& tgt_mask) -> torch::Tensor {
            // src: [batch_size, src_len]
            // tgt: [batch_size, tgt_len]
            // return: [batch_size, tgt_len, tgt_vocab_size]
            auto memory = encoder(src, src_mask); // [batch_size, src_len, model_dim]
            auto output = decoder(memory, src_mask, tgt, tgt_mask); // [batch_size, tgt_len, model_dim]
            //output = lm_head(output); // [batch_size, tgt_len, tgt_vocab_size]
            return output;
        }
    };
    TORCH_MODULE(TransformerNMT);

}
