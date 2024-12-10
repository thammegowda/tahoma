#include <tahoma/utils.h>
#include  <tahoma/model/mt5.h>

using namespace tahoma;
namespace tahoma::model::mt5 {


    auto LayerNormImpl::forward(Tensor x) -> Tensor {
        auto variance = x.to(torch::kFloat32).pow(2).mean(-1, true);
        x = x / (variance + variance_epsilon).sqrt();
        if (weight.dtype() != x.dtype()) {
            x = x.to(weight.dtype());
        }
        return x * weight;
    }

    auto GELUNewImpl::forward(Tensor x) -> Tensor {
        return 0.5 * x * (1.0 + torch::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * torch::pow(x, 3.0))));
    }

    DenseGatedActDenseImpl::DenseGatedActDenseImpl(size_t model_dim, size_t ff_dim, double dropout) :
        wi_0{ register_module("wi_0", nn::Linear(nn::LinearOptions(model_dim, ff_dim).bias(false))) },
        wi_1{ register_module("wi_1", nn::Linear(nn::LinearOptions(model_dim, ff_dim).bias(false))) },
        wo{ register_module("wo", nn::Linear(nn::LinearOptions(ff_dim, model_dim).bias(false))) },
        dropout{ register_module("dropout", nn::Dropout(dropout)) },
        act{ register_module("act", GELUNew()) }
    {
    }

    auto DenseGatedActDenseImpl::forward(Tensor x) -> Tensor {
        auto hidden_gelu = act(wi_0(x));
        auto hidden_linear = wi_1(x);
        x = hidden_gelu * hidden_linear;
        x = dropout(wo(x));
        return x;
    }

    FFSubLayerImpl::FFSubLayerImpl(size_t model_dim, size_t ff_dim, double dropout, size_t layer_idx) :
        layer_norm{ register_module("layer_norm", LayerNorm(model_dim)) },
        DenseReluDense{ register_module("DenseReluDense", DenseGatedActDense(model_dim, ff_dim, dropout)) },
        dropout{ register_module("dropout", nn::Dropout(dropout)) },
        layer_idx{ layer_idx }
    {
    }

    auto FFSubLayerImpl::forward(Tensor x) -> Tensor {
        auto y = layer_norm(x);
        y = DenseReluDense(y);
        y = dropout(y);
        auto z = x + y;
        return z;
    }

    AttentionImpl::AttentionImpl(const YAML::Node& config, bool is_bidirectional, bool has_relative_attention_bias) :
        d_model{ config["d_model"].as<i32>() },
        n_heads{ config["num_heads"].as<i32>() },
        d_kv{ d_model / n_heads },
        inner_dim{ d_kv * n_heads },
        is_bidirectional{ is_bidirectional },
        has_relative_attention_bias{ has_relative_attention_bias },
        relative_attention_num_buckets{ config["relative_attention_num_buckets"].as<i32>(DEF_REL_POS_BUCKETS) },
        relative_attention_max_distance{ config["relative_attention_max_distance"].as<i32>(DEF_REL_POS_MAX_DISTANCE) },
        q{ register_module("q", nn::Linear(nn::LinearOptions(d_model, inner_dim).bias(false))) },
        k{ register_module("k", nn::Linear(nn::LinearOptions(d_model, inner_dim).bias(false))) },
        v{ register_module("v", nn::Linear(nn::LinearOptions(d_model, inner_dim).bias(false))) },
        o{ register_module("o", nn::Linear(nn::LinearOptions(inner_dim, d_model).bias(false))) },
        dropout{ register_module("dropout", nn::Dropout(config["dropout_rate"].as<float>())) }
    {
        if (has_relative_attention_bias) {
            relative_attention_bias = register_module("relative_attention_bias", nn::Embedding(nn::EmbeddingOptions(relative_attention_num_buckets, n_heads)));
        }
    }

    auto AttentionImpl::relative_position_bucket(Tensor relative_position) -> Tensor {
        /**
         *  Adapted from Mesh Tensorflow:
        * https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer (e.g. 32)
            max_distance: an integer (e.g. 128)

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        */
        //auto rel_buckets = 0;
        auto num_buckets = relative_attention_num_buckets;
        auto max_distance = relative_attention_max_distance;
        if (num_buckets < 0 || max_distance < 0 || max_distance % 2 != 0) {
            throw std::runtime_error("num_buckets and max_distance must be positive. num_buckets must be even");
        }
        auto relative_buckets = torch::zeros_like(relative_position);
        if (is_bidirectional) {
            num_buckets /= 2;
            relative_buckets += (relative_position > 0).to(torch::kLong) * num_buckets;
            relative_position = torch::abs(relative_position);
        }
        else {
            // zero out future positions for auto-regressive decoder
            relative_position = -torch::min(relative_position, torch::zeros_like(relative_position, relative_position.dtype()));
        }
        // if bucket size is 32, and bidirectional, then, upto 16 are for left context
        // half of the buckets are for exact increments in positions
        // 0 1 2 3 4 5 6 7 8 <-- exact increments
        //   then 8 to 16 are in log scale increments
        i64 max_exact = num_buckets / 2;
        auto is_small = relative_position < max_exact;
        // ask chatgpt to parse this expression and explain the order of operations
        auto rel_pos_if_large = max_exact + (torch::log(relative_position.to(torch::kFloat) / max_exact) / std::log((float)max_distance / (float)max_exact) * (num_buckets - max_exact)).to(torch::kLong);
        rel_pos_if_large = torch::min(rel_pos_if_large,
            torch::full_like(rel_pos_if_large, num_buckets - 1));

        relative_buckets += torch::where(is_small, relative_position, rel_pos_if_large);
        return relative_buckets;
    }

    auto AttentionImpl::compute_bias(i64 query_length, i64 mem_length) -> Tensor {
        auto qry_pos = torch::arange(query_length, torch::kInt64).view({ -1, 1 });
        auto mem_pos = torch::arange(mem_length, torch::kInt64).view({ 1, -1 });
        auto rel_pos = mem_pos - qry_pos;
        auto rel_pos_bucket = relative_position_bucket(rel_pos);
        auto pos_embs = relative_attention_bias(rel_pos_bucket);
        pos_embs = pos_embs.permute({ 2, 0, 1 }).unsqueeze(0);
        return pos_embs;
    }

    auto AttentionImpl::forward(Pack& pack) -> Pack {
        Tensor query = std::any_cast<Tensor>(pack["query"]);
        Tensor docs = std::any_cast<Tensor>(pack["docs"]);
        Tensor mask = std::any_cast<Tensor>(pack["mask"]);
        Tensor position_bias;
        if (pack.contains("position_bias")) {
            position_bias = std::any_cast<Tensor>(pack["position_bias"]);
        }
        Pack output = {};

        auto batch_size = query.size(0);
        auto qs = q(query).view({ batch_size, -1, n_heads, d_kv }).transpose(1, 2);
        auto ks = k(docs).view({ batch_size, -1, n_heads, d_kv }).transpose(1, 2);
        auto vs = v(docs).view({ batch_size, -1, n_heads, d_kv }).transpose(1, 2);

        auto attn_weights = torch::matmul(qs, ks.transpose(3, 2));

        // position_bias gets computed on the first layer and reused in subsequent layers
        if (!position_bias.defined() && has_relative_attention_bias) {
            i64 n_queries = query.size(1);
            i64 n_docs = docs.size(1);
            position_bias = compute_bias(n_queries, n_docs);
            output["position_bias"] = position_bias;
        }
        if (position_bias.defined()) {
            attn_weights += position_bias;
        }

        if (mask.defined()) {
            attn_weights = attn_weights.masked_fill(mask, -1e9);
        }
        attn_weights = F::softmax(attn_weights, -1);
        attn_weights = dropout(attn_weights);
        auto attn_out = torch::matmul(attn_weights, vs);
        attn_out = attn_out.transpose(1, 2).contiguous().view({ batch_size, -1, n_heads * d_kv });
        attn_out = o(attn_out);
        output["result"] = attn_out;
        return output;
    }

    SelfAttentionImpl::SelfAttentionImpl(const YAML::Node& config, bool has_relative_attention_bias, size_t layer_idx) :
        SelfAttention{ register_module("SelfAttention",
                Attention(config,  /*bidirectional*/true, has_relative_attention_bias)) },
        layer_norm{ register_module("layer_norm", LayerNorm(config["d_model"].as<i32>())) },
        dropout{ register_module("dropout", nn::Dropout(config["dropout_rate"].as<float>())) },
        layer_idx{ layer_idx }
    {
    }

    auto SelfAttentionImpl::forward(Pack& pack) -> Pack {
        /*
        layer_norm -> attention -> dropout -> residual_connection
        but forward any other outputs from attention layer (e.g. position_bias)
        */
        auto x = std::any_cast<Tensor>(pack["input"]);
        auto y = layer_norm(x);\
            Pack args = {
                {"query", y},
                {"docs", y},
                {"mask", pack["input_mask"]}
        };
        if (pack.contains("position_bias")) {
            args["position_bias"] = pack["position_bias"];
        }
        auto attn_out = SelfAttention(args);
        y = std::any_cast<Tensor>(attn_out["result"]);
        attn_out["result"] = x + dropout(y);
        return attn_out;
    }

    CrossAttentionImpl::CrossAttentionImpl(const YAML::Node& config, size_t layer_idx) :
        EncDecAttention{ register_module("EncDecAttention",
         Attention(config, /*bidirectional=*/false, /*relative_attention_bias*/ false)) },
        layer_norm{ register_module("layer_norm", LayerNorm(config["d_model"].as<i32>())) },
        dropout{ register_module("dropout", nn::Dropout(config["dropout_rate"].as<float>())) },
        layer_idx{ layer_idx }
    {
    }

    auto CrossAttentionImpl::forward(Pack& pack) -> Pack {
        auto x = std::any_cast<Tensor>(pack["input"]);
        auto y = layer_norm(x);
        Pack args = {
            {"query", y},
            {"docs", pack["memory"]},
            {"mask", pack["memory_mask"]}
        };
        auto attn_out = EncDecAttention(args);
        y = std::any_cast<Tensor>(attn_out["result"]);
        attn_out["result"] = x + dropout(y);
        return attn_out;
    }

    BlockImpl::BlockImpl(const YAML::Node& config, bool has_relative_attention_bias,
        bool is_decoder, size_t layer_idx) :
        is_decoder{ is_decoder },
        layer{ register_module("layer", nn::ModuleList()) },
        layer_idx{ layer_idx }

    {
        layer->push_back(SelfAttention(config, has_relative_attention_bias, layer_idx));
        if (is_decoder) {
            layer->push_back(CrossAttention(config, layer_idx));
        }
        auto model_dim = config["d_model"].as<size_t>();
        auto ff_dim = config["d_ff"].as<size_t>();
        auto dropout = config["dropout_rate"].as<float>();
        layer->push_back(FFSubLayer(model_dim, ff_dim, dropout, layer_idx));
    }

    auto BlockImpl::forward(Pack& pack) -> Pack {
        Pack args = pack;
        Pack temp;

        size_t idx = 0;
        auto self_atttn = std::dynamic_pointer_cast<SelfAttentionImpl>(layer[idx++]);
        temp = self_atttn->forward(args);
        args["input"] = temp["result"];
        if (temp.contains("position_bias")) {
            args["position_bias"] = temp["position_bias"];
        }

        if (is_decoder) {
            auto memory = std::any_cast<Tensor>(args["memory"]);
            auto memory_mask = std::any_cast<Tensor>(args["memory_mask"]);
            auto src_atttn = std::dynamic_pointer_cast<CrossAttentionImpl>(layer[idx++]);
            temp = src_atttn->forward(args);
            args["input"] = temp["result"];
        }

        auto input = std::any_cast<Tensor>(args["input"]);
        auto ff_layer = std::dynamic_pointer_cast<FFSubLayerImpl>(layer[idx++]);
        args["result"] = ff_layer->forward(input);
        args.erase("input");
        return args;
    }

    StackImpl::StackImpl(const YAML::Node& config, bool is_decoder) :
        is_decoder{ is_decoder },
        block{ register_module("block", nn::ModuleList()) },
        final_layer_norm{ register_module("final_layer_norm", LayerNorm(config["d_model"].as<size_t>())) },
        dropout{ register_module("dropout", nn::Dropout(config["dropout_rate"].as<float>())) }
    {
        size_t num_layers = config[is_decoder ? "num_decoder_layers" : "num_layers"].as<size_t>();
        size_t model_dim = config["d_model"].as<size_t>();
        size_t ff_dim = config["d_ff"].as<size_t>();
        size_t n_heads = config["num_heads"].as<size_t>();
        float dropout_rate = config["dropout_rate"].as<float>();

        for (int idx = 0; idx < num_layers; idx++) {
            bool has_rel_pos_emb = idx == 0; // only the first layer has relative position embedding
            block->push_back(Block(config, has_rel_pos_emb, is_decoder, idx));
        }
    }

    auto StackImpl::forward(Pack& args) -> Pack {
        Pack temp;
        size_t idx = 0;
        for (const auto& module : *block) {
            temp = std::dynamic_pointer_cast<BlockImpl>(module)->forward(args);
            args["input"] = temp["result"];
            // we need to pass position_bias to the next layer
            if (idx == 0 && temp.contains("position_bias")) {
                args["position_bias"] = temp["position_bias"];
            }
            idx++;
        }
        Tensor hidden_states = std::any_cast<Tensor>(args["input"]);
        hidden_states = final_layer_norm(hidden_states);
        hidden_states = dropout(hidden_states);
        return { {"result", hidden_states} };
    }

    ConditionalGenerationImpl::ConditionalGenerationImpl(const YAML::Node& config) :
        LanguageModel(config["d_model"].as<size_t>(), config["vocab_size"].as<size_t>(), /*lm_bias*/false),
        shared{ register_module("shared", nn::Embedding(nn::EmbeddingOptions(vocab_size, model_dim).padding_idx(0))) },
        encoder{ register_module("encoder", Stack(config, /*is_decoder=*/false)) },
        decoder{ register_module("decoder", Stack(config, /*is_decoder=*/true)) } {
        auto feed_forward_proj = config["feed_forward_proj"].as<std::string>();
        if (feed_forward_proj != "gated-gelu") {
            throw std::runtime_error("Only gated-gelu is supported for feed_forward_proj");
        }
    }

    auto ConditionalGenerationImpl::forward(Pack& args) -> Pack {
        auto src = std::any_cast<Tensor>(args["src"]);
        auto tgt = std::any_cast<Tensor>(args["tgt"]);
        auto src_mask = std::any_cast<Tensor>(args["src_mask"]);
        auto tgt_mask = std::any_cast<Tensor>(args["tgt_mask"]);

        auto src_emb = shared(src);
        Pack enc_args = {
            {"input", src_emb},
            {"input_mask", src_mask}
        };
        auto enc_out = encoder(enc_args);
        if (enc_out.find("result") != enc_out.end()) {
            std::runtime_error("Encoder forward must return a Pack with key 'result'");
        }
        auto memory = std::any_cast<Tensor>(enc_out["result"]);
        Pack dec_args = {
            {"input", shared(tgt)},
            {"input_mask", tgt_mask},
            {"memory", memory},
            {"memory_mask", src_mask}
        };
        auto output = std::any_cast<Tensor>(decoder(dec_args)["result"]);
        output = lm_head(output);
        return { {"result", output} };
    }

    auto ConditionalGenerationImpl::greedy_decode(Tensor input, Tensor input_mask,
        i32 bos_id, i32 eos_id, i32 max_new_toks) -> Tensor {
        auto input_embs = shared(input);
        Pack args = {
            {"input", input_embs},
            {"input_mask", input_mask}
        };
        auto out = encoder(args);
        auto memory = out["result"];
        auto batch_size = input.size(0);

        auto tgt_seq = torch::full({ batch_size, 1 }, bos_id,
            torch::device(input.device()).dtype(torch::kLong));
        auto tgt_embds = shared(tgt_seq);
        auto tgt_mask = torch::zeros({ batch_size, 1, 1, 1 },
            torch::device(input.device()).dtype(torch::kBool));

        auto src_len = input.size(1);
        auto max_tgt_len = src_len + max_new_toks;
        Pack dec_args = {
            {"memory", memory},
            {"memory_mask", input_mask}
        };
        Tensor dec_out, last_state, last_logits;
        for (auto i = 0; i < max_tgt_len; i++) {
            dec_args["input"] = tgt_embds;
            dec_args["input_mask"] = tgt_mask;
            dec_out = std::any_cast<Tensor>(decoder(dec_args)["result"]);

            last_state = dec_out.index({ Slice(), -1, Slice() }); // B x d_model
            last_logits = lm_head(last_state); // B x vocab_size
            auto last_token = last_logits.argmax(-1, true); // B x 1
            tgt_seq = torch::cat({ tgt_seq, last_token }, 1);
            tgt_embds = shared(tgt_seq);
            tgt_mask = torch::zeros_like(tgt_seq, torch::device(tgt_seq.device()).dtype(torch::kBool)).view({ batch_size, 1, 1, -1 });
        }
        return tgt_seq;
    }

}