#pragma once
#include <sentencepiece_processor.h>
#include <tahoma.h>
#include <tahoma/model/transformer_nmt.h>

 namespace tahoma::inference {

    class Decoder {
    private:
        std::shared_ptr<model::TransformerNMTImpl> _model;
        nn::AnyModule _lm_head;
        std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> _vocabs;
        torch::Device _device;

    public:
        Decoder(std::shared_ptr<model::TransformerNMTImpl> model,
                nn::AnyModule lm_head,
                std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> vocabs,
                torch::Device device);

        auto greedy_decode(str src, i32 max_len = 128) -> std::tuple<str, f32>;
    };
}
