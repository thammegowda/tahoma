#pragma once
#include <tahoma.h>
#include <tahoma/model.h>
#include <tahoma/layer/transformer.h>


namespace tahoma::model::xlmr {

    struct XLMRoberaEmbeddingsImpl : public nn::Module {
        i64 vocab_size;
        i64 hidden_size;
        i64 type_vocab_size;
        i64 max_position_embeddings;
        string position_embedding_type;
        f64 dropout_rate;
        i64 pad_token_id;
        nn::Embedding word_embeddings;
        nn::Embedding position_embeddings;
        nn::Embedding token_type_embeddings;
        nn::Dropout dropout;
        //Tensor position_ids;
        //Tensor token_type_ids;

        XLMRoberaEmbeddingsImpl(Pack& args) :
            vocab_size{ args.get<i64>("vocab_size") },
            hidden_size{ args.get<i64>("hidden_size") },
            type_vocab_size{ args.get<i64>("type_vocab_size") },
            max_position_embeddings{ args.get<i64>("max_position_embeddings") },
            dropout_rate{ args.get<f64>("dropout_prob") },
            position_embedding_type{ args.get<>("position_embedding_type", "absolute") },
            pad_token_id{ args.get<i64>("pad_token_id", 1) },

            word_embeddings{ register_module("word_embeddings",
                 nn::Embedding(nn::EmbeddingOptions(vocab_size, hidden_size).padding_idx(pad_token_id))) },
            position_embeddings{ register_module("position_embedding",
                 nn::Embedding(nn::EmbeddingOptions(max_position_embeddings, hidden_size).padding_idx(pad_token_id))) },
            token_type_embeddings{ register_module("token_type_embeddings",
                 nn::Embedding(type_vocab_size, hidden_size)) },
            dropout{ register_module("dropout", nn::Dropout(dropout_rate)) }/*,
            position_ids{ register_buffer("position_ids",
                torch::arange(max_position_embeddings).expand({1, -1})) },
            token_type_ids{ register_buffer("token_type_ids",
                torch::zeros({1, max_position_embeddings}, torch::dtype(torch::kLong))) }*/

        {
        }

        auto forward(Tensor input_ids) -> torch::Tensor{
            auto seq_length = input_ids.size(1);
            auto shift = pad_token_id +1;
            auto position_ids = torch::arange(shift, shift + seq_length,
                 torch::device(input_ids.device()).dtype(torch::kLong)).expand({1, -1}).to(input_ids.device());
            auto words_embs = word_embeddings(input_ids);
            auto pos_embs = this->position_embeddings(position_ids);
            auto type_embs = torch::zeros_like(position_ids);
            return dropout(words_embs + pos_embs + type_embs);
        }
    };
    TORCH_MODULE(XLMRoberaEmbeddings);


    class XLMREncoderImpl : public nn::Module {
    };
    TORCH_MODULE(XLMREncoder);

    class XCOMETImpl : public nn::Module {

        XLMREncoder encoder;
        


    };
    TORCH_MODULE(XCOMET);

}