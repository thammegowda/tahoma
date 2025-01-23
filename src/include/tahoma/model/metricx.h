#pragma once

#include <tahoma/model/mt5.h>
#include <tahoma/autocast.h>


using namespace tahoma;

namespace tahoma::model::metricx {

    struct RegressionImpl : public model::mt5::ConditionalGenerationImpl {

        const i32 cls_tok_id = 250089; // <extra_id_10>
        const string model_id;
        const i32 model_year;
        bool tie_word_embeddings = false;

        RegressionImpl(const YAML::Node& config)
            : model::mt5::ConditionalGenerationImpl(config),
            model_id{ config["model_id"].as<string>() },
            model_year{ find_model_year(model_id) },
            tie_word_embeddings{ config["tie_word_embeddings"].as<bool>() }
        {}

        auto task_type() -> TaskType override {
            return TaskType::REGRESSION;
        }

        auto find_model_year(string model_id) -> i32 {
            if (model_id.find("metricx-23") != string::npos) {
                return 2023;
            } else if (model_id.find("metricx-24") != string::npos) {
                return 2024;
            }
            throw std::runtime_error("Unsupported model year");
        }
        auto forward(Pack& args) -> Pack {
            auto input = std::any_cast<Tensor>(args["input"]);
            auto input_mask = std::any_cast<Tensor>(args["input_mask"]);
            auto src_emb = shared(input); // [B X T X D]
            Pack enc_args = {
                {"input", src_emb},
                {"input_mask", input_mask}
            };
            auto memory = encoder(enc_args).get<Tensor>("result");
            auto tgt_seq = torch::zeros({ input.size(0), 1 }, torch::dtype(torch::kLong).device(input.device()));
            Pack dec_args = {
                {"input", shared(tgt_seq)},
                {"input_mask", torch::ones_like(tgt_seq).to(torch::kBool)},
                {"memory", memory},
                {"memory_mask", input_mask}
            };

            // decoder values go out of fp16 range and cause NaNs, we autocast encoder and not for decoder
            AutoCastGuard guard(src_emb.device().type(), false);
            auto dec_out = decoder(dec_args).get<Tensor>("result");
            if (tie_word_embeddings) {
            // # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                dec_out = dec_out * (std::pow(model_dim, -0.5));
            }
            dec_out = dec_out.index({ Slice(), 0, Slice() }); // [B x D]  // the first token repr
            auto final_weights = lm_head->weight.index({cls_tok_id, Slice()}).unsqueeze(1);  // [D x 1]
            auto predictions = torch::matmul(dec_out, final_weights).squeeze(1);  // [BxD] [Dx1] = [Bx1] = [B]
            predictions = torch::clamp(predictions, 0, 25);
            return { {"result", predictions} };
        }

        auto make_input(std::map<string, string> example, bool is_qe) -> string {
            string input;

            if (model_year == 2023) {
                input = "candidate: " + example["candidate"];
                input += is_qe ? " source: " + example["source"] : " reference: " + example["reference"];
            } else if (model_year == 2024) {
                input = "source: " + example["source"] + " candidate: " + example["candidate"];
                if (!is_qe) {
                    input += " reference: " + example["reference"];
                }
            } else {
                throw std::runtime_error("Unsupported model year");
            }
            return input;
        }

        auto max_input_length() -> size_t {
            switch (model_year) {
            case 2023:
                return 1024;
            case 2024:
                return 1536;
            default:
                throw std::runtime_error("Unsupported model year");
            }
        }

    };
    TORCH_MODULE(Regression);
}