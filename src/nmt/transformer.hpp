#include <iostream>
#include <torch/torch.h>
//#include <toml++/toml.h>
#include "../common/config.hpp"


namespace nn = torch::nn;
namespace F = torch::nn::functional;
using namespace rtg;
using namespace torch::indexing;


namespace rtg::nmt::transformer {



    struct PositionEmbeddingImpl : nn::Module {
        nn::Embedding embedding;
        torch::Tensor positions;
        nn::Dropout dropout;


        PositionEmbeddingImpl(int vocab_size, int model_dim, double dropout = 0.1, int max_len = 5000) :
            embedding{ register_module("embedding", nn::Embedding(nn::EmbeddingOptions(vocab_size, model_dim))) },
            dropout{ register_module("dropout", nn::Dropout(nn::DropoutOptions(dropout))) },
            positions{ torch::zeros({1, max_len, model_dim}, torch::requires_grad(false)) }
        {

            /* python ::
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                */

            auto pos = torch::arange(max_len, torch::kLong);         // [max_len]
            auto div_term = torch::exp(torch::arange(0, model_dim, 2) * (-std::log(10'000.0) / model_dim));
            positions.index_put_({ Slice(), Slice(), Slice(0, None, 2) }, torch::sin(pos.unsqueeze(1) * div_term));
            positions.index_put_({ Slice(), Slice(), Slice(1, None, 2) }, torch::cos(pos.unsqueeze(1) * div_term));
            //positions = positions.unsqueeze(0);                           // [1, max_len, model_dim]
        }

        auto forward(torch::Tensor& x) -> torch::Tensor {
            x = embedding(x);   // [batch_size, seq_len] -> [batch_size, seq_len, model_dim]
            // x = x+pe[:, :x.size(1)]
            auto pe = positions.index({ Slice(), Slice(None, x.size(1)) });
            x = x + pe;
            if (is_training()) {
                x = dropout(x);
            }
            return x;
        }
    };
    TORCH_MODULE(PositionEmbedding);

    struct TransformerNMTImpl : public nn::Module {
        PositionEmbedding src_embed;
        PositionEmbedding tgt_embed;
        nn::Transformer transformer;

        TransformerNMTImpl(const rtg::config::Config& config)
            :src_embed{ register_module("src_embed", PositionEmbedding(
                    config["model"]["src_vocab_size"].as<int>(),
                    config["model"]["model_dim"].as<int>(),
                    config["model"]["dropout"].as<double>())) },
            tgt_embed{ register_module("tgt_embed", PositionEmbedding(
               config["model"]["tgt_vocab_size"].as<int>(),
               config["model"]["model_dim"].as<int>(),
               config["model"]["dropout"].as<double>())) },
            transformer{ register_module("transformer", init_model(config)) }
        {
        }

        static auto init_model(config::Config config) -> nn::Transformer {
            if (!config["model"]) {
                spdlog::error("[model] config block not found");
                throw std::runtime_error("[model] config block not found");
            }
            auto model_conf = config["model"];

            // check if model_dim, nhead, num_encoder_layers, num_decoder_layers are present
            auto required_keys = { "model_dim", "attn_head", "encoder_layers", "decoder_layers", "dropout", "ffn_dim" };
            for (auto& key : required_keys) {
                if (!model_conf[key]) {
                    spdlog::error("model config: {} not found", key);
                    throw std::runtime_error("model config: " + std::string(key) + " not found");
                }
            }
            // value<T>() returns optional<T>, and then value() returns T
            auto d_model = model_conf["model_dim"].as<int64_t>(512);
            auto nhead = model_conf["attn_head"].as<int64_t>(8);
            auto num_encoder_layers = model_conf["encoder_layers"].as<int64_t>(6);
            auto num_decoder_layers = model_conf["decoder_layers"].as<int64_t>(6);
            auto dropout = model_conf["dropout"].as<double>(0.1);
            auto ffn_dim = model_conf["ffn_dim"].as<double>(d_model * 4);

            auto model_config = nn::TransformerOptions(d_model, nhead)
                .num_encoder_layers(num_encoder_layers)
                .dim_feedforward(ffn_dim)
                .dropout(dropout)
                .num_decoder_layers(num_decoder_layers);
            return nn::Transformer(model_config);
        }

        auto forward(torch::Tensor& src, torch::Tensor& tgt, torch::Tensor& src_mask, torch::Tensor& tgt_mask) -> torch::Tensor {
            // src: [batch_size, src_len]
            // tgt: [batch_size, tgt_len]
            // return: [batch_size, tgt_len, tgt_vocab_size]

            std::cout << "src: " << src.sizes() << std::endl;
            std::cout << "tgt: " << tgt.sizes() << std::endl;
            auto src_embedded = src_embed(src); // [batch_size, src_len, model_dim]
            auto tgt_embedded = tgt_embed(tgt); // [batch_size, tgt_len, model_dim]
            std::cout << "src_embedded: " << src_embedded.sizes() << std::endl;
            std::cout << "tgt_embedded: " << tgt_embedded.sizes() << std::endl;
            std::cout << "src_mask: " << src_mask.sizes() << std::endl;
            std::cout << "tgt_mask: " << tgt_mask.sizes() << std::endl;
            auto memory = transformer(src_embedded, tgt_embedded, src_mask, tgt_mask);
            return memory;
        }
    };
    TORCH_MODULE(TransformerNMT);


}
