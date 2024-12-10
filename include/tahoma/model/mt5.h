#pragma once
#include <tahoma.h>
#include <tahoma/model.h>
#include <tahoma/layer/transformer.h>

using namespace tahoma;
namespace tahoma::model::mt5 {

    inline bool is_debugging = false;
    inline i32 DEF_REL_POS_BUCKETS = 32;
    inline i32 DEF_REL_POS_MAX_DISTANCE = 128;

    struct LayerNormImpl : public nn::Module {
        Tensor weight;
        float variance_epsilon;
        LayerNormImpl(size_t hidden_size, float eps = 1e-6) :
            weight{ register_parameter("weight", torch::ones(hidden_size)) },
            variance_epsilon{ eps } {
        }

        auto forward(Tensor x) -> Tensor;
    };
    TORCH_MODULE(LayerNorm);

    class GELUNewImpl : public nn::Module {
    public:
        GELUNewImpl() {};
        auto forward(Tensor x) -> Tensor;
    };
    TORCH_MODULE(GELUNew);

    struct DenseGatedActDenseImpl : public nn::Module {
        nn::Linear wi_0, wi_1, wo;
        nn::Dropout dropout;
        GELUNew act;
        DenseGatedActDenseImpl(size_t model_dim, size_t ff_dim, double dropout = 0.0);
        auto forward(Tensor x) -> Tensor;
    };
    TORCH_MODULE(DenseGatedActDense);

    struct FFSubLayerImpl : public nn::Module {
        DenseGatedActDense DenseReluDense;
        LayerNorm layer_norm;
        nn::Dropout dropout;
        size_t layer_idx;
        FFSubLayerImpl(size_t model_dim, size_t ff_dim, double dropout_rate = 0.0, size_t layer_idx = 0);
        auto forward(Tensor x) -> Tensor;
    };
    TORCH_MODULE(FFSubLayer);

    struct AttentionImpl : public nn::Module {
        i32 d_model;
        i32 n_heads;
        i32 d_kv;
        bool is_bidirectional;
        i32 inner_dim;
        i32 relative_attention_num_buckets;
        i32 relative_attention_max_distance;
        nn::Linear q, k, v, o;
        bool has_relative_attention_bias;
        nn::Embedding relative_attention_bias = nullptr; // this can be null if has_relative_attention_bias=false
        nn::Dropout dropout;
        AttentionImpl(const YAML::Node& config, bool is_bidirectional, bool has_relative_attention_bias);
        auto relative_position_bucket(Tensor relative_position) -> Tensor;
        auto compute_bias(i64 query_length, i64 mem_length) -> Tensor;
        auto forward(Pack& pack) -> Pack;
    };
    TORCH_MODULE(Attention);

    struct SelfAttentionImpl : public nn::Module {
        Attention SelfAttention;
        LayerNorm layer_norm;
        nn::Dropout dropout;
        size_t layer_idx;
        SelfAttentionImpl(const YAML::Node& config, bool has_relative_attention_bias, size_t layer_idx);
        auto forward(Pack& pack) -> Pack;
    };
    TORCH_MODULE(SelfAttention);

    struct CrossAttentionImpl : public nn::Module {
        Attention EncDecAttention;
        LayerNorm layer_norm;
        nn::Dropout dropout;
        size_t layer_idx;
        CrossAttentionImpl(const YAML::Node& config, size_t layer_idx);
        auto forward(Pack& pack) -> Pack;
    };
    TORCH_MODULE(CrossAttention);

    struct BlockImpl : public nn::Module {
        bool is_decoder;
        nn::ModuleList layer;  // list of sub-layers: self-attention, cross-attention, feed-forward
        size_t layer_idx;
        BlockImpl(const YAML::Node& config, bool has_relative_attention_bias,
            bool is_decoder, size_t layer_idx = 0);
        auto forward(Pack& args) -> Pack;
    };
    TORCH_MODULE(Block);

    struct StackImpl : public nn::Module {
        bool is_decoder;
        nn::ModuleList block; // List of Blocks
        LayerNorm final_layer_norm;
        nn::Dropout dropout;
        StackImpl(const YAML::Node& config, bool is_decoder = false);
        auto forward(Pack& args) -> Pack;
    };
    TORCH_MODULE(Stack);

    struct ConditionalGenerationImpl : public LanguageModel {
        nn::Embedding shared;
        Stack encoder;
        Stack decoder;
        ConditionalGenerationImpl(const YAML::Node& config);
        auto task_type() -> TaskType override { return TaskType::NMT; } 
        auto forward(Pack& args) -> Pack;

       auto greedy_decode(Tensor input, Tensor input_mask, i32 bos_id, i32 eos_id, i32 max_new_toks) -> Tensor;

    };

    TORCH_MODULE(ConditionalGeneration);

}