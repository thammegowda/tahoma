#pragma once

#include "../train/utils.hpp"
#include "../model/transformer_nmt.hpp"



namespace rtg::inference {

    
    class Decoder {
    private:
        rtg::model::TransformerNMT _model;
        nn::AnyModule _lm_head;
        vector<shared_ptr<sp::SentencePieceProcessor>> _vocabs;
        torch::Device _device;
    public:
        Decoder(rtg::model::TransformerNMT _model, nn::AnyModule lm_head, vector<shared_ptr<sp::SentencePieceProcessor>> _vocabs, torch::Device _device):
            _model {_model}, _lm_head {lm_head}, _vocabs{_vocabs}, _device{_device}
        {
            if (_vocabs.size() != 2){
                throw std::invalid_argument("Vocab size must be 2, but found " + _vocabs.size());
            }
        }

        auto greedy_decode(str src, i32 max_len=128) -> str{
            auto src_vocab = _vocabs[0];
            auto tgt_vocab = _vocabs[1];
            vector<int> src_ids_vec = _vocabs[0]->EncodeAsIds(src);
            auto src_ids = torch::tensor(src_ids_vec, torch::dtype(torch::kInt64).device(_device)).unsqueeze(0); // [1, src_len]

            auto src_mask = (src_ids == src_vocab->pad_id()).view({1, 1, 1, -1}); // [batch=1, 1, 1, src_len]
            src_mask = src_mask.to(torch::kBool);
            auto memory = _model ->encoder(src_ids, src_mask);            
            auto tgt_ids = torch::full({src_ids.size(0), 1}, tgt_vocab->bos_id(), torch::dtype(torch::kInt64).device(_device));
            for (int i=0; i < max_len; i++){
                auto tgt_len = tgt_ids.size(1);
                auto tgt_mask = rtg::train::subsequent_mask(tgt_len, _device).to(torch::kBool).view({1, 1, tgt_len, tgt_len});  // [batch=1, head=1, tgt_len, tgt_len]
                auto features = _model->decoder(memory, src_mask, tgt_ids, tgt_mask);
                features = features.index({Slice(), -1, Slice()});
                auto output = _lm_head.forward(features);
                auto next_token = output.view({1, -1}).argmax(-1);
                tgt_ids = torch::cat({tgt_ids, next_token.view({1, -1})}, 1);
                // TODO: max and compute score
                // TODO: Halt on EOS
            }
            // convert torch Tensor into cpp vector<int64>
            tgt_ids = tgt_ids.view({-1}).to(torch::kCPU).contiguous();
            std::vector<i64> tgt_ids_vec(tgt_ids.data_ptr<i64>(), tgt_ids.data_ptr<i64>() + tgt_ids.numel());
            std::vector<int> tgt_ids_vec2(tgt_ids_vec.begin(), tgt_ids_vec.end());
            auto tgt_tokens = _vocabs[1]->DecodeIds(tgt_ids_vec2);  // spm takes int and not int64
            return tgt_tokens;
        }
    };
}