#include <iostream>
#include <torch/torch.h>


namespace nn = torch::nn;

namespace rtg::nmt::transformer {

    auto init_model(toml::table config) -> nn::Transformer {
        if (!config["model"]) {
            spdlog::error("[model] config block not found");
            throw std::runtime_error("[model] config block not found");
        }
        auto model_conf = config["model"];

        // check if model_dim, nhead, num_encoder_layers, num_decoder_layers are present
        auto required_keys = { "model_dim", "attn_head", "encoder_layers", "decoder_layers", "dropout", "ffn_dim" };
        for (auto& key : required_keys) {
            if (!model_conf.as_table()->contains(key)) {
                spdlog::error("model config: {} not found", key);
                throw std::runtime_error("model config: " + std::string(key) + " not found");
            }
        }
        // value<T>() returns optional<T>, and then value() returns T
        auto d_model = model_conf["model_dim"].value_or<int64_t>(512);
        auto nhead = model_conf["attn_head"].value_or<int64_t>(8);
        auto num_encoder_layers = model_conf["encoder_layers"].value_or<int64_t>(6);
        auto num_decoder_layers = model_conf["decoder_layers"].value_or<int64_t>(6);
        auto dropout = model_conf["dropout"].value_or<double>(0.1);
        auto ffn_dim = model_conf["ffn_dim"].value_or<double>(d_model * 4);

        auto model_config = nn::TransformerOptions(d_model, nhead)
            .num_encoder_layers(num_encoder_layers)
            .dim_feedforward(ffn_dim)
            .dropout(dropout)
            .num_decoder_layers(num_decoder_layers);
        auto model = nn::Transformer(model_config);
        std::cout << model << "\n";
        return model;
    }
}
