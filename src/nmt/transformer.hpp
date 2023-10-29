#include <iostream>
#include <tuple>
#include <assert.h>
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

    struct EncoderLayerImpl : public nn::Module {
        nn::MultiheadAttention self_attn;
        nn::Linear fc1;
        nn::Linear fc2;
        nn::LayerNorm norm1;
        nn::LayerNorm norm2;
        nn::Dropout dropout1;
        nn::Dropout dropout2;

        EncoderLayerImpl(int model_dim, int nhead, double dropout = 0.1) :
           
            self_attn{
                (assert(model_dim > 0), assert(nhead > 0), assert(model_dim % nhead == 0),
                register_module("self_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(model_dim, nhead).dropout(dropout))))
                },
            fc1{ register_module("fc1", nn::Linear(nn::LinearOptions(model_dim, model_dim * 4))) },
            fc2{ register_module("fc2", nn::Linear(nn::LinearOptions(model_dim * 4, model_dim))) },
            norm1{ register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            norm2{ register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            dropout1{ register_module("dropout1", nn::Dropout(nn::DropoutOptions(dropout))) },
            dropout2{ register_module("dropout2", nn::Dropout(nn::DropoutOptions(dropout))) }
        {
        }

        auto forward(torch::Tensor& src, torch::Tensor& src_mask) -> torch::Tensor {
            // src: [batch_size, src_len, model_dim]
            // src_mask: [batch_size, 1, src_len]
            // return: [batch_size, src_len, model_dim]

            auto src2 = std::get<0>(
                self_attn(src, src, src, src_mask)); // [batch_size, src_len, model_dim]
            src = src + dropout1(src2);
            src = norm1(src);

            src2 = fc2(F::relu(fc1(src))); // [batch_size, src_len, model_dim]
            src = src + dropout2(src2);
            src = norm2(src);
            return src;
        }
    };
    TORCH_MODULE(EncoderLayer);


    struct EncoderImpl : public nn::Module {
        PositionEmbedding position_embedding = nullptr;
        nn::LayerNorm norm1;
        nn::Dropout dropout;
        int num_layers;
        nn::ModuleList layers;

        static nn::ModuleList make_layers(int model_dim, int nhead, int num_layers, double dropout = 0.1) {
            auto layers = nn::ModuleList(); 
            for (int i = 0; i < num_layers; ++i) {
                layers->push_back(EncoderLayer(model_dim, nhead, dropout));
            }
            return layers;
        }

        EncoderImpl(int vocab_size, int model_dim, int nhead, int num_layers, double dropout = 0.1) :
            //embedding{ register_module("embedding", nn::Embedding(nn::EmbeddingOptions(vocab_size, model_dim))) },
            position_embedding{ register_module("position_embedding", PositionEmbedding(vocab_size, model_dim, dropout)) },
            norm1{ register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            dropout{ register_module("dropout", nn::Dropout(nn::DropoutOptions(dropout))) },
            layers{ register_module("layers", make_layers(model_dim, nhead, num_layers, dropout))},
            num_layers{ num_layers }
        {
        }

        auto forward(torch::Tensor& src, torch::Tensor& src_mask) -> torch::Tensor {
            // src: [batch_size, src_len]
            // src_mask: [batch_size, 1, src_len]
            // return: [batch_size, src_len, model_dim]

            auto x = src;
            //auto src_embedded = embedding(src); // [batch_size, src_len, model_dim]
            x = position_embedding(x); // [batch_size, src_len, model_dim]
            x = dropout(x);
            x = norm1(x);
            
            for (int i = 0; i < num_layers; ++i) {
                auto layer = layers->at<EncoderLayerImpl>(i);
                x = layer.forward(x, src_mask);
            }
            return x;
        }
    };
    TORCH_MODULE(Encoder);


    struct DecoderLayerImpl : public nn::Module {

        nn::MultiheadAttention self_attn;
        nn::MultiheadAttention src_attn;
        nn::Linear fc1;
        nn::Linear fc2;

        nn::LayerNorm norm1;
        nn::LayerNorm norm2;
        nn::LayerNorm norm3;

        nn::Dropout dropout1;
        nn::Dropout dropout2;
        nn::Dropout dropout3;


        DecoderLayerImpl(int model_dim, int nhead, double dropout = 0.1) :
            self_attn {( 
                assert(model_dim > 0), 
                assert(nhead > 0),
                assert(model_dim % nhead == 0), 
                register_module("self_attn",
                    nn::MultiheadAttention(nn::MultiheadAttentionOptions(model_dim, nhead).dropout(dropout)))
            )},
            src_attn {
                register_module("src_attn", 
                    nn::MultiheadAttention(nn::MultiheadAttentionOptions(model_dim, nhead).dropout(dropout))) 
            },
            fc1 { register_module("fc1", nn::Linear(nn::LinearOptions(model_dim, model_dim * 4))) },
            fc2 { register_module("fc2", nn::Linear(nn::LinearOptions(model_dim * 4, model_dim))) },
            norm1 { register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            norm2 { register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            norm3 { register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            dropout1 { register_module("dropout1", nn::Dropout(nn::DropoutOptions(dropout))) },
            dropout2 { register_module("dropout2", nn::Dropout(nn::DropoutOptions(dropout))) },
            dropout3 { register_module("dropout3", nn::Dropout(nn::DropoutOptions(dropout))) }
        {
        }

        auto forward(torch::Tensor& tgt, torch::Tensor& memory, torch::Tensor& tgt_mask, torch::Tensor& memory_mask) -> torch::Tensor {
            // tgt: [batch_size, tgt_len, model_dim]
            // memory: [batch_size, src_len, model_dim]
            // tgt_mask: [batch_size, 1, tgt_len]
            // memory_mask: [batch_size, 1, src_len]
            // return

            auto x = tgt;
            auto x2 = std::get<0>(self_attn(x, x, x, tgt_mask)); // [batch_size, tgt_len, model_dim]
            x = x + dropout1(x2);
            x = norm1(x);

            x2 = std::get<0>(src_attn(x, memory, memory, memory_mask)); // [batch_size, tgt_len, model_dim]
            x = x + dropout2(x2);
            x = norm2(x);

            x2 = fc2(F::relu(fc1(x))); // [batch_size, tgt_len, model_dim]
            x = x + dropout3(x2);
            x = norm3(x);
            return x;
        }
    };
    TORCH_MODULE(DecoderLayer);

    struct DecoderImpl: public nn::Module {
        PositionEmbedding position_embedding;
        nn::LayerNorm norm1;
        nn::Dropout dropout;
        int num_layers;
        nn::ModuleList layers;

        DecoderImpl(int vocab_size, int model_dim, int nhead, int num_layers, double dropout = 0.1) :
            position_embedding{( assert(vocab_size > 0), assert(model_dim > 0),
                register_module("position_embedding", PositionEmbedding(vocab_size, model_dim, dropout)) 
                )},
            norm1{ register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({ model_dim }))) },
            dropout{ register_module("dropout", nn::Dropout(nn::DropoutOptions(dropout))) },
            layers{ register_module("layers", nn::ModuleList()) },
            num_layers{ num_layers }
        {
            assert(num_layers > 0);
            for (int i = 0; i < num_layers; ++i) {
                layers->push_back(DecoderLayer(model_dim, nhead, dropout));
            }
        }

        auto forward(torch::Tensor& tgt, torch::Tensor& memory, torch::Tensor& tgt_mask, torch::Tensor& memory_mask) -> torch::Tensor {
            // tgt: [batch_size, tgt_len]
            // memory: [batch_size, src_len, model_dim]
            // tgt_mask: [batch_size, 1, tgt_len]
            // memory_mask: [batch_size, 1, src_len]
            // return: [batch_size, tgt_len, model_dim]

            auto x = tgt;
            x = position_embedding(x); // [batch_size, tgt_len, model_dim]
            x = dropout(x);
            x = norm1(x);

            for (int i = 0; i < num_layers; ++i) {
                x = layers->at<DecoderLayerImpl>(i).forward(x, memory, tgt_mask, memory_mask);
            }
            return x;
        }
    };
    TORCH_MODULE(Decoder);


    struct TransformerNMTImpl : public nn::Module {
        Encoder encoder;
        Decoder decoder;

        TransformerNMTImpl(const rtg::config::Config& config):
            encoder {
                register_module("encoder", Encoder(
                config["model"]["src_vocab_size"].as<int>(),
                config["model"]["model_dim"].as<int>(),
                config["model"]["attn_head"].as<int>(),
                config["model"]["encoder_layers"].as<int>(),
                config["model"]["dropout"].as<double>()))
            },
            decoder {
                register_module("decoder",  Decoder(
                config["model"]["tgt_vocab_size"].as<int>(),
                config["model"]["model_dim"].as<int>(),
                config["model"]["attn_head"].as<int>(),
                config["model"]["decoder_layers"].as<int>(),
                config["model"]["dropout"].as<double>()))
            }
        {
        }

        auto forward(torch::Tensor& src, torch::Tensor& tgt, torch::Tensor& src_mask, torch::Tensor& tgt_mask) -> torch::Tensor {
            // src: [batch_size, src_len]
            // tgt: [batch_size, tgt_len]
            // return: [batch_size, tgt_len, tgt_vocab_size]

            std::cout << "src: " << src.sizes() << std::endl;
            std::cout << "tgt: " << tgt.sizes() << std::endl;

            auto memory = encoder(src, src_mask); // [batch_size, src_len, model_dim]
            auto output = decoder(tgt, memory, tgt_mask, src_mask); // [batch_size, tgt_len, model_dim]
            return output;
        }
    };
    TORCH_MODULE(TransformerNMT);

}
