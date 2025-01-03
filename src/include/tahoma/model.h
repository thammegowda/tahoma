
#pragma once

#include <torch/torch.h>
#include <tahoma.h>

using namespace tahoma;

namespace tahoma::model {

    struct IModel: public nn::Module {
            IModel() = default;
            ~IModel() = default;

            virtual TaskType task_type() = 0;
            // virtual functions and templates dont mix. so we use std::any for the return type
            virtual Pack forward(Pack& args) = 0;

            virtual std::string name(){
                return typeid(*this).name();
            }

            auto get_state() -> Pack;
            auto set_state(Pack& state) -> Pack;
        };


    struct LanguageModel: public IModel {
        size_t model_dim;
        size_t vocab_size;
        nn::Linear lm_head;

        LanguageModel(size_t model_dim, size_t vocab_size, bool lm_bias=false):
            model_dim { model_dim },
            vocab_size { vocab_size },
            lm_head { register_module("lm_head", 
                nn::Linear(nn::LinearOptions(model_dim, vocab_size).bias(false))) }
        {}
    };

}