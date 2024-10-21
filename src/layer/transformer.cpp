#include <tuple>
#include <cmath>
#include <iostream>
#include <tuple>
#include <assert.h>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>

#include <tahoma.h>
#include <tahoma/layer/transformer.h>


namespace nn = torch::nn;
namespace F = torch::nn::functional;
using namespace tahoma;
using namespace torch::indexing;

namespace tahoma::layer::transformer {


    auto MultiheadAttentionImpl::forward(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor key_padding_mask)
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
                std::string _shape = "";
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



    auto TransformerEncoderLayerImpl::forward(torch::Tensor src, torch::Tensor src_mask) -> torch::Tensor {
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

    auto TransformerEncoderImpl::forward(torch::Tensor src, torch::Tensor src_mask) -> torch::Tensor {
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

    auto TransformerDecoderLayerImpl::forward(torch::Tensor tgt, torch::Tensor tgt_mask, torch::Tensor memory, torch::Tensor memory_mask) -> torch::Tensor {
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


    auto TransformerDecoderImpl::forward(torch::Tensor tgt, torch::Tensor tgt_mask, torch::Tensor memory, torch::Tensor memory_mask) -> torch::Tensor {
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


}
