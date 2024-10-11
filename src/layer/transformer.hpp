#pragma once

#include <tuple>
#include <cmath>
#include <iostream>
#include <tuple>
#include <assert.h>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include "../common/config.hpp"


namespace nn = torch::nn;
namespace F = torch::nn::functional;
using namespace tahoma;
using namespace torch::indexing;


namespace tahoma::layer {

    // TODO: split this into a separate separate position+word embeddings
    struct AbsolutePositionEmbeddingImpl : nn::Module {
        nn::Embedding embedding;
        torch::Tensor positions;
        nn::Dropout dropout;
        int model_dim;

        AbsolutePositionEmbeddingImpl(int vocab_size, int model_dim, double dropout = 0.1, const int max_len = 5000) :
            model_dim{ model_dim },
            embedding{ register_module("embedding", nn::Embedding(nn::EmbeddingOptions(vocab_size, model_dim))) },
            dropout{ register_module("dropout", nn::Dropout(nn::DropoutOptions(dropout))) },
            positions{ torch::zeros({1, max_len, model_dim}, torch::requires_grad(false)) }
        {
            /* python ::
                pe = torch.zeros(1, max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
                pe[:, :, 0::2] = torch.sin(position * div_term)
                pe[:, :, 1::2] = torch.cos(position * div_term)  */
            auto pos = torch::arange(max_len, torch::kLong);         // [max_len]
            auto div_term = torch::exp(torch::arange(0, model_dim, 2) * (-std::log(10'000.0) / model_dim));
            positions.index_put_({ Slice(), Slice(), Slice(0, None, 2) }, torch::sin(pos.unsqueeze(1) * div_term));
            positions.index_put_({ Slice(), Slice(), Slice(1, None, 2) }, torch::cos(pos.unsqueeze(1) * div_term));
            register_buffer("positions", positions);  //required for this->to(device) to work
        }

        auto forward(torch::Tensor x) -> torch::Tensor {
            x = embedding(x);   // [batch_size, seq_len] -> [batch_size, seq_len, model_dim]
            x = x * std::sqrt(model_dim);
            // x = x + pe[:, :x.size(1)]
            auto pe = positions.index({ Slice(), Slice(None, x.size(1)) });
            x = x + pe;
            x = dropout(x);
            return x;
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
        {
        }

        auto forward(torch::Tensor x) -> torch::Tensor {
            x = F::gelu(fc1(x));
            x = dropout(fc2(x));
            return x;
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

        auto forward(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor key_padding_mask)
            -> std::pair<torch::Tensor, torch::Tensor> {
            // query:   [batch_size, tgt_len, model_dim]
            // key, val: [batch_size, src_len, model_dim]
            // key_padding_mask: [batch_size, 1, T, src_len]
            // return: [batch_size, tgt_len, model_dim], [batch_size, tgt_len, src_len]

            auto tgt_len = query.size(1);
            auto src_len = key.size(1);
            auto batch_size = query.size(0);
            assert(batch_size == key.size(0));

            assert(query.size(2) == model_dim);
            assert(key.size(2) == model_dim);
            assert(value.size(2) == model_dim);
            auto head_dim = model_dim / nhead;
            auto q = q_proj(query).view({ batch_size, tgt_len, nhead, head_dim }).transpose(1, 2); // [batch_size, nhead, tgt_len, head_dim]
            auto k = k_proj(key).view({ batch_size, src_len, nhead, head_dim }).transpose(1, 2); // [batch_size, nhead, src_len, head_dim]
            auto v = v_proj(value).view({ batch_size, src_len, nhead, head_dim }).transpose(1, 2); // [batch_size, nhead, src_len, head_dim]
            auto attn_weights = torch::matmul(q, k.transpose(-2, -1)); // [batch_size, nhead, tgt_len, src_len]
            attn_weights = attn_weights / std::sqrt(head_dim);

            if (key_padding_mask.defined()) {
                // insert head dim
                //key_padding_mask = key_padding_mask.unsqueeze(1); // [batch_size, 1, 1, src_len]
                if (key_padding_mask.sizes().size() != 4){ // must be a 4D tensor
                    string _shape = "";
                    for (auto i : key_padding_mask.sizes()) { _shape += std::to_string(i) + ", "; }
                    throw std::runtime_error("key_padding_mask must be a 4D tensor. given: [" +  _shape + "]");
                }
                float low_val = -1e9;
                if (at::autocast::is_autocast_enabled(query.device().type())){
                    low_val = -pow(2, 14);   // -16384.0f; TODO: check for bf16
                }
                attn_weights = attn_weights.masked_fill(key_padding_mask, low_val);
            }
            attn_weights = F::softmax(attn_weights, -1); // [batch_size, nhead, tgt_len, src_len]
            attn_weights = dropout(attn_weights);
            auto attn_output = torch::matmul(attn_weights, v) // [batch_size, nhead, tgt_len, head_dim]
                                .transpose(1, 2)              // [batch_size, tgt_len, nhead, head_dim]
                                .contiguous().view({ batch_size, tgt_len, model_dim }); // [batch_size, tgt_len, model_dim]

            attn_output = out_proj(attn_output); // [batch_size, tgt_len, model_dim]

            return std::make_pair(attn_output, attn_weights);
        }
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
            norm1{ ( // using comma operator to assert multiple conditions before first initialization
                assert(model_dim > 0),
                assert(nhead > 0),
                assert(model_dim % nhead == 0),
                assert(ffn_dim > 0),
                register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim }) )))},
            self_attn{register_module("self_attn", MultiheadAttention(model_dim, nhead, dropout))},
            dropout1{ register_module("dropout1", nn::Dropout(nn::DropoutOptions(dropout))) },

            norm2{ register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            ffn{ register_module("ffn", FFSubLayer(model_dim, ffn_dim, dropout)) },
            dropout2{ register_module("dropout2", nn::Dropout(nn::DropoutOptions(dropout))) }
        {}

        auto forward(torch::Tensor src, torch::Tensor src_mask) -> torch::Tensor {
            // src: [batch_size, src_len, model_dim]
            // src_mask: [batch_size, 1, src_len]
            // return: [batch_size, src_len, model_dim]

            auto x = norm1(src);
            x = self_attn(x, x, x, src_mask).first; // [batch_size, src_len, model_dim]
            src = src + dropout1(x);

            x = norm2(src);
            x = ffn(x);
            src = src + dropout2(x);
            return src;
        }
    };
    TORCH_MODULE(TransformerEncoderLayer);


    struct TransformerEncoderImpl : public nn::Module {
        AbsolutePositionEmbedding position_embedding;
        nn::LayerNorm norm;
        int num_layers;
        nn::ModuleList layers;

        TransformerEncoderImpl(int vocab_size, int model_dim, int ffn_dim, int nhead, int num_layers, double dropout = 0.1) :
            //embedding{ register_module("embedding", nn::Embedding(nn::EmbeddingOptions(vocab_size, model_dim))) },
            num_layers{ num_layers },
            position_embedding{ register_module("position_embedding", AbsolutePositionEmbedding(vocab_size, model_dim, dropout)) },
            layers{ register_module("layers", nn::ModuleList())},
            norm{ register_module("norm", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) }
        {
             for (int i = 0; i < num_layers; i++) {
                layers->push_back(TransformerEncoderLayer(model_dim, ffn_dim, nhead, dropout));
            }
        }

        auto forward(torch::Tensor src, torch::Tensor src_mask) -> torch::Tensor {
            // src: [batch_size, src_len]
            // src_mask: [batch_size, 1, src_len]
            // return: [batch_size, src_len, model_dim]

            auto x = position_embedding(src); // [batch_size, src_len, model_dim]
            // x = dropout(x);  dropout already applied by position_embedding

            for (int i = 0; i < num_layers; ++i) {
                auto layer = layers->at<TransformerEncoderLayerImpl>(i);
                x = layer.forward(x, src_mask);
            }
            x = norm(x);
            return x;
        }
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
            norm1 { ( // using comma operator to assert multiple conditions before first initialization
                assert(model_dim > 0),
                assert(nhead > 0),
                assert(model_dim % nhead == 0),
                assert(ffn_dim > 0),
                register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim })))) },
            self_attn {register_module("self_attn", MultiheadAttention(model_dim, nhead, dropout))},
            dropout1 { register_module("dropout1", nn::Dropout(nn::DropoutOptions(dropout))) },

            norm2 { register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            src_attn { register_module("src_attn", MultiheadAttention(model_dim, nhead, dropout)) },
            dropout2 { register_module("dropout2", nn::Dropout(nn::DropoutOptions(dropout))) },

            norm3 { register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            ffn { register_module("ffn", FFSubLayer(model_dim, ffn_dim, dropout)) },
            dropout3 { register_module("dropout3", nn::Dropout(nn::DropoutOptions(dropout))) }
        {
        }

        auto forward(torch::Tensor tgt, torch::Tensor tgt_mask, torch::Tensor memory, torch::Tensor memory_mask) -> torch::Tensor {
            // tgt: [batch_size, tgt_len, model_dim]
            // memory: [batch_size, src_len, model_dim]
            // tgt_mask: [batch_size, 1, tgt_len]
            // memory_mask: [batch_size, 1, src_len]
            // return: [batch_size, tgt_len, model_dim]

            // Self attention sublayer
            torch::Tensor x = norm1(tgt);
            x = self_attn(x, x, x, tgt_mask).first; // [batch_size, tgt_len, model_dim]
            tgt = tgt + dropout1(x);

            // Source attention sublayer
            x = norm2(tgt);
            x = src_attn(x, memory, memory, memory_mask).first; // [batch_size, tgt_len, model_dim]
            tgt = tgt + dropout2(x);

            // Feed forward sublayer
            x = norm3(tgt);
            x = ffn(x);
            tgt = tgt + dropout3(x);
            return tgt;
        }
    };
    TORCH_MODULE(TransformerDecoderLayer);

    struct TransformerDecoderImpl: public nn::Module {
        int num_layers;
        int model_dim;
        int nhead;

        AbsolutePositionEmbedding position_embedding;
        nn::LayerNorm norm;
        nn::ModuleList layers;

        TransformerDecoderImpl(int vocab_size, int model_dim, int ffn_dim, int nhead, int num_layers, double dropout = 0.1) :
            model_dim {( assert(model_dim > 0),  model_dim )},
            num_layers {( assert(num_layers > 0), num_layers )},
            nhead {( assert(nhead >0), assert(model_dim % nhead == 0),  nhead )},
            position_embedding{(
                assert(vocab_size > 0), assert(model_dim > 0),
                register_module("position_embedding", AbsolutePositionEmbedding(vocab_size, model_dim, dropout)) 
                )},
            norm{ register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            layers{ register_module("layers", nn::ModuleList()) }
        {
            for (int i = 0; i < num_layers; ++i) {
                layers->push_back(TransformerDecoderLayer(model_dim, ffn_dim, nhead, dropout));
            }
        }

        auto forward(torch::Tensor tgt, torch::Tensor tgt_mask, torch::Tensor memory, torch::Tensor memory_mask) -> torch::Tensor {
            // tgt: [batch_size, tgt_len]
            // memory: [batch_size, src_len, model_dim]
            // tgt_mask: [batch_size, 1, tgt_len]
            // memory_mask: [batch_size, 1, src_len]
            // return: [batch_size, tgt_len, model_dim]

            auto x = position_embedding(tgt); // [batch_size, tgt_len, model_dim]
            //x = dropout(x); already applied by position_embedding

            for (int i = 0; i < num_layers; i++) {
                x = layers->at<TransformerDecoderLayerImpl>(i).forward(x, tgt_mask, memory, memory_mask);
            }
            //x = lm_head(x); // [batch_size, tgt_len, vocab_size]
            x = norm(x);
            return x;
        }
    };
    TORCH_MODULE(TransformerDecoder);


}
