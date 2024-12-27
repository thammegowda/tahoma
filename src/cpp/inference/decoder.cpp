
#include <tahoma.h>
#include <tahoma/model/transformer_nmt.h>
#include <tahoma/inference/decoder.h>
#include <tahoma/utils.h>


namespace sp = sentencepiece;
using namespace tahoma;
using namespace torch::indexing;

namespace tahoma::inference {

    Decoder::Decoder(std::shared_ptr<model::LanguageModel> _model,
        nn::AnyModule lm_head,
        vector<std::shared_ptr<sp::SentencePieceProcessor>> _vocabs,
        torch::Device _device) :
        _model{ _model }, _lm_head{ lm_head }, _vocabs{ _vocabs }, _device{ _device } {
       
        if (_vocabs.size() != 2) {
            throw std::invalid_argument("Vocab size must be 2, but found " + _vocabs.size());
        }
    }

    auto Decoder::greedy_decode(std::string src, i32 max_len) -> std::tuple<std::string, f32> {
        auto src_vocab = _vocabs[0];
        auto tgt_vocab = _vocabs[1];
        vector<int> src_ids_vec = _vocabs[0]->EncodeAsIds(src);
        auto src_ids = torch::tensor(src_ids_vec, torch::dtype(torch::kInt64).device(_device)).unsqueeze(0); // [1, src_len]

        auto src_mask = (src_ids == src_vocab->pad_id()).view({ 1, 1, 1, -1 }); // [batch=1, 1, 1, src_len]
        src_mask = src_mask.to(torch::kBool);

        // TODO: support other model types; only NMT model is supported as of now
         if (_model->task_type() != TaskType::NMT) {
            throw std::invalid_argument("Only NMT model is supported");
        }
        auto model = std::dynamic_pointer_cast<model::TransformerNMTImpl>(_model);

        auto memory = model->encoder(src_ids, src_mask);
        auto tgt_ids = torch::full({ src_ids.size(0), 1 }, tgt_vocab->bos_id(), torch::dtype(torch::kInt64).device(_device));
        f32 total_score = 0.0;
        for (int i = 0; i < max_len; i++) {
            auto tgt_len = tgt_ids.size(1);
            auto tgt_mask = utils::subsequent_mask(tgt_len, _device).to(torch::kBool).view({ 1, 1, tgt_len, tgt_len });  // [batch=1, head=1, tgt_len, tgt_len]
            auto features = model->decoder(tgt_ids, tgt_mask, memory, src_mask);
            features = features.index({ Slice(), -1, Slice() });
            auto output = _lm_head.forward(features);
            //auto next_token = output.view({1, -1}).argmax(-1);
            // TODO: max and compute score
            auto [best_score, best_token] = output.log_softmax(-1).max(-1);
            total_score += best_score.item<float>();
            tgt_ids = torch::cat({ tgt_ids, best_token.view({1, -1}) }, 1);
            if (best_token.item<int64_t>() == tgt_vocab->eos_id()) {
                break;
            }
        }
        // convert torch Tensor into cpp vector<int64>
        tgt_ids = tgt_ids.view({ -1 }).to(torch::kCPU).contiguous();
        std::vector<i64> tgt_ids_vec(tgt_ids.data_ptr<i64>() + 1, tgt_ids.data_ptr<i64>() + tgt_ids.numel());
        std::vector<int> tgt_ids_vec2(tgt_ids_vec.begin(), tgt_ids_vec.end());
        //string ids_str = ""; for (auto i: tgt_ids_vec2) ids_str += std::to_string(i) + " ";
        //spdlog::info("HYP IDs: {}", ids_str);
        auto tgt_tokens = _vocabs[1]->DecodeIds(tgt_ids_vec2);  // spm takes int and not int64
        auto avg_score = total_score / tgt_ids_vec.size();
        return { tgt_tokens, avg_score };
    }
}