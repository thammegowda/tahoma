#pragma once
#include <torch/torch.h>
#include <tahoma.h>

namespace nn = torch::nn;
namespace F = torch::nn::functional;
using namespace tahoma;

namespace tahoma::layer::transformer {

    struct AbsolutePositionEmbeddingImpl : nn::Module {
        int model_dim;
        nn::Dropout dropout;
        nn::Embedding embedding;
        torch::Tensor positions;

        AbsolutePositionEmbeddingImpl(int vocab_size, int model_dim, double dropout = 0.1, const int max_len = 5000) :
            model_dim{ model_dim },
            dropout{ register_module("dropout", nn::Dropout(nn::DropoutOptions(dropout))) },
            embedding{ register_module("embedding", nn::Embedding(nn::EmbeddingOptions(vocab_size, model_dim))) },
            positions{ torch::zeros({1, max_len, model_dim}, torch::requires_grad(false)) }
        {
            auto pos = torch::arange(max_len, torch::kLong);
            auto div_term = torch::exp(torch::arange(0, model_dim, 2) * (-std::log(10'000.0) / model_dim));
            positions.index_put_({ Slice(), Slice(), Slice(0, None, 2) }, torch::sin(pos.unsqueeze(1) * div_term));
            positions.index_put_({ Slice(), Slice(), Slice(1, None, 2) }, torch::cos(pos.unsqueeze(1) * div_term));
            register_buffer("positions", positions);
        }

        auto forward(torch::Tensor x) -> torch::Tensor {
            x = embedding(x) * std::sqrt(model_dim) + positions.index({ Slice(), Slice(None, x.size(1)) });
            return dropout(x);
        }
    };
    TORCH_MODULE(AbsolutePositionEmbedding);

    struct FFSubLayerImpl : public nn::Module {
        nn::Linear fc1;
        nn::Linear fc2;
        nn::Dropout dropout;

        FFSubLayerImpl(int model_dim, int ff_dim, double dropout = 0.1) :
            fc1{ register_module("fc1", nn::Linear(nn::LinearOptions(model_dim, ff_dim))) },
            fc2{ register_module("fc2", nn::Linear(nn::LinearOptions(ff_dim, model_dim))) },
            dropout{ register_module("dropout", nn::Dropout(nn::DropoutOptions(dropout))) }
        {}

        auto forward(torch::Tensor x) -> torch::Tensor {
            x = F::gelu(fc1(x));
            return dropout(fc2(x));
        }
    };
    TORCH_MODULE(FFSubLayer);

    struct MultiheadAttentionImpl: public nn::Module {
        nn::Linear q_proj;
        nn::Linear k_proj;
        nn::Linear v_proj;
        nn::Linear out_proj;
        nn::Dropout dropout;
        int model_dim;
        int nhead;

        MultiheadAttentionImpl(int model_dim, int nhead, double dropout = 0.1) :
            q_proj{ (assert(model_dim > 0), assert(nhead > 0), assert(model_dim % nhead == 0),
                register_module("q_proj", nn::Linear(nn::LinearOptions(model_dim, model_dim)))) },
            k_proj{ register_module("k_proj", nn::Linear(nn::LinearOptions(model_dim, model_dim))) },
            v_proj{ register_module("v_proj", nn::Linear(nn::LinearOptions(model_dim, model_dim))) },
            out_proj{ register_module("out_proj", nn::Linear(nn::LinearOptions(model_dim, model_dim))) },
            dropout{ register_module("dropout", nn::Dropout(nn::DropoutOptions(dropout))) },
            model_dim{ model_dim },
            nhead{ nhead }
        {}

        auto forward(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor key_mask)
            -> std::pair<torch::Tensor, torch::Tensor>;
    };
    TORCH_MODULE(MultiheadAttention);

    struct TransformerEncoderLayerImpl : public nn::Module {
        nn::LayerNorm norm1;
        MultiheadAttention self_attn;
        nn::Dropout dropout1;

        nn::LayerNorm norm2;
        FFSubLayer ffn;
        nn::Dropout dropout2;

        TransformerEncoderLayerImpl(int model_dim, int ffn_dim, int nhead, double dropout = 0.1) :
            norm1{ (assert(model_dim > 0), assert(nhead > 0), assert(model_dim % nhead == 0), assert(ffn_dim > 0),
                register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim })))) },
            self_attn{ register_module("self_attn", MultiheadAttention(model_dim, nhead, dropout)) },
            dropout1{ register_module("dropout1", nn::Dropout(nn::DropoutOptions(dropout))) },
            norm2{ register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            ffn{ register_module("ffn", FFSubLayer(model_dim, ffn_dim, dropout)) },
            dropout2{ register_module("dropout2", nn::Dropout(nn::DropoutOptions(dropout))) }
        {}

        auto forward(torch::Tensor src, torch::Tensor src_mask) -> torch::Tensor;
    };
    TORCH_MODULE(TransformerEncoderLayer);

    struct TransformerEncoderImpl : public nn::Module {
        int num_layers;
        AbsolutePositionEmbedding position_embedding;
        nn::LayerNorm norm;
        nn::ModuleList layers;

        TransformerEncoderImpl(int vocab_size, int model_dim, int ffn_dim, int nhead, int num_layers, double dropout = 0.1) :
            num_layers{ num_layers },
            position_embedding{ register_module("position_embedding", AbsolutePositionEmbedding(vocab_size, model_dim, dropout)) },
            layers{ register_module("layers", nn::ModuleList()) },
            norm{ register_module("norm", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) }
        {
            for (int i = 0; i < num_layers; i++) {
                layers->push_back(TransformerEncoderLayer(model_dim, ffn_dim, nhead, dropout));
            }
        }

        auto forward(torch::Tensor src, torch::Tensor src_mask) -> torch::Tensor;
    };
    TORCH_MODULE(TransformerEncoder);

    struct TransformerDecoderLayerImpl : public nn::Module {
        nn::LayerNorm norm1;
        MultiheadAttention self_attn;
        nn::Dropout dropout1;

        nn::LayerNorm norm2;
        MultiheadAttention src_attn;
        nn::Dropout dropout2;

        nn::LayerNorm norm3;
        FFSubLayer ffn;
        nn::Dropout dropout3;

        TransformerDecoderLayerImpl(int model_dim, int ffn_dim, int nhead, double dropout = 0.1) :
            norm1{ (assert(model_dim > 0), assert(nhead > 0), assert(model_dim % nhead == 0), assert(ffn_dim > 0),
                register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim })))) },
            self_attn{ register_module("self_attn", MultiheadAttention(model_dim, nhead, dropout)) },
            dropout1{ register_module("dropout1", nn::Dropout(nn::DropoutOptions(dropout))) },
            norm2{ register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            src_attn{ register_module("src_attn", MultiheadAttention(model_dim, nhead, dropout)) },
            dropout2{ register_module("dropout2", nn::Dropout(nn::DropoutOptions(dropout))) },
            norm3{ register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            ffn{ register_module("ffn", FFSubLayer(model_dim, ffn_dim, dropout)) },
            dropout3{ register_module("dropout3", nn::Dropout(nn::DropoutOptions(dropout))) }
        {}

        auto forward(torch::Tensor tgt, torch::Tensor tgt_mask, torch::Tensor memory, torch::Tensor memory_mask) -> torch::Tensor;
    };
    TORCH_MODULE(TransformerDecoderLayer);

    struct TransformerDecoderImpl: public nn::Module {
        int num_layers;
        int model_dim;
        int nhead;

        AbsolutePositionEmbedding position_embedding;
        nn::LayerNorm norm;
        nn::ModuleList layers;

        TransformerDecoderImpl(int vocab_size, int model_dim, int ffn_dim, int nhead,
                int num_layers, double dropout = 0.1) :
            num_layers{ (assert(num_layers > 0), num_layers) },
            model_dim{ (assert(model_dim > 0), model_dim) },
            nhead{ (assert(nhead > 0), assert(model_dim % nhead == 0), nhead) },
            position_embedding{ (assert(vocab_size > 0), assert(model_dim > 0),
                register_module("position_embedding", AbsolutePositionEmbedding(vocab_size, model_dim, dropout))) },
            norm{ register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            layers{ register_module("layers", nn::ModuleList()) }
        {
            for (int i = 0; i < num_layers; i++) {
                layers->push_back(TransformerDecoderLayer(model_dim, ffn_dim, nhead, dropout));
            }
        }

        auto forward(torch::Tensor tgt, torch::Tensor tgt_mask, torch::Tensor memory, torch::Tensor memory_mask) -> torch::Tensor;
    };
    TORCH_MODULE(TransformerDecoder);

} // namespace tahoma::layer::transformer