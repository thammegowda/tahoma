
#include <tahoma.h>
#include <tahoma/inference/predictor.h>
#include <tahoma/utils.h>
#include <tahoma/vocab.h>
#include <tahoma/data.h>
#include <tahoma/model/metricx.h>
#include <tahoma/serialize.h>

namespace tahoma::inference {

    void predict_scores(Ptr<model::LanguageModel> model, data::DataLoader& loader, string file_name, Pack kwargs) {
        bool is_qe = kwargs.get("qe", false);
        size_t width = kwargs.get("width", 6);
        size_t batch_size = kwargs.get("batch_size", 4);

        auto lines = utils::read_lines(file_name);
        assertm(model->task_type() == TaskType::REGRESSION, "Only regression models are supported for scoring");
        // TODO: support other models
        auto regression_model = std::dynamic_pointer_cast<model::metricx::RegressionImpl>(model);
        if (regression_model == nullptr) {
            throw std::runtime_error("Failed to cast model to RegressionImpl");
        }

        bool print_model = kwargs.get("print_model", false);
        if (print_model) {
            std::cerr << "model:\n" << *model << std::endl;
        }

        auto PAD_ID = loader.vocabs[0]->pad_id();
        torch::Device device = DEVICE;
        vector<i32> eos_ids = {}; // EOS IDs are not required. Google's code removes it from HF tokenizer's output
        vector<size_t> max_lens = { 1024 };

        regression_model->eval();
        regression_model->to(device);


        auto make_mask = [&](const torch::Tensor& x) -> torch::Tensor {
            // assume x is [BxT]; return [Bx1x1xT]; model requires 4D mask
            return x.eq(PAD_ID).to(torch::kBool).to(x.device()).view({ x.size(0), 1, 1, x.size(1) });
            };


        auto consume_buffer = [&](vector2d<string> buffer) -> void {
            // consume buffer
            vector<data::Example> examples;
            for (auto& line : buffer) {
                auto example = loader.make_example(line, eos_ids, max_lens, /*max_length_crop=*/true);
                examples.push_back(example);
            }
            auto batch = data::Batch(examples, /*contiguous=*/true);
            batch.to(device);
            Pack inps = {
                {"input", batch.fields[0]},
                {"input_mask", make_mask(batch.fields[0])},
            };

            auto out = model->forward(inps);
            auto width = 6;
            auto scores = std::any_cast<torch::Tensor>(out["result"]);
            for (int i = 0; i < scores.size(0); ++i) {
                std::cout << fmt::format("{:.{}f}", scores[i].item<float>(), width) << "\t" << examples[i].fields[0] << std::endl;
            }
            };

        vector2d<string> buffer;
        for (auto& line : lines) {
            auto parts = utils::split(line, "\t");
            size_t expected_fields = is_qe ? 2 : 3;
            if (parts.size() < expected_fields) {
                throw std::runtime_error("Invalid input line: " + line + " Expected at least "
                    + std::to_string(expected_fields) + " fields but got " + std::to_string(parts.size()));
            }
            std::map<string, string> example = {
                {"source", parts[0]},
                {"candidate", parts[1]},
            };
            if (!is_qe) {
                example["reference"] = parts[2];
            }
            auto input_line = regression_model->make_input(example, is_qe);
            buffer.push_back({ input_line });
            if (buffer.size() >= batch_size) {
                consume_buffer(buffer);
                buffer.clear();
            }
        }
        if (!buffer.empty()) {
            consume_buffer(buffer);
        }
    }

    void decode(Ptr<model::LanguageModel> model, data::DataLoader& loader, string file_name, Pack kwargs) {
        size_t batch_size = 4; //def
        if (kwargs.contains("batch_size")) {
            batch_size = std::any_cast<size_t>(kwargs["batch_size"]);
        }

        auto lines = utils::read_lines(file_name);
        assertm(model->task_type() == TaskType::NMT, "Only NMT models are supported for decoding");
        auto nmt_model = std::dynamic_pointer_cast<model::mt5::ConditionalGenerationImpl>(model);
        if (nmt_model == nullptr) {
            throw std::runtime_error("Failed to cast model to ConditionalGenerationImpl");
        }

        auto EOS_ID = loader.vocabs[0]->eos_id();
        auto PAD_ID = loader.vocabs[0]->pad_id();
        vector<i32> eos_ids = { EOS_ID, EOS_ID };
        vector<size_t> max_lens = { 512, 512 };
        i32 max_new_toks = 64;
        auto tgt_vocab = loader.vocabs[loader.vocabs.size() > 1 ? 1 : 0];

        auto make_mask = [&](const torch::Tensor& x) -> torch::Tensor {
            // assume x is [BxT]; return [Bx1x1xT]; model requires 4D mask
            return x.eq(PAD_ID).unsqueeze(1).unsqueeze(2).to(torch::kBool).to(x.device());
            };

        auto consume_buffer = [&](vector2d<string> buffer) -> void {
            vector<data::Example> examples;
            for (auto& line : buffer) {
                auto example = loader.make_example(line, eos_ids, max_lens, /*max_length_crop=*/true);
                examples.push_back(example);
            }
            auto batch = data::Batch(examples, /*contiguous=*/true);
            auto batch_size = buffer.size();
            auto input = batch.fields[0];
            auto input_mask = make_mask(batch.fields[0]);
            auto tgt_seq_ids = nmt_model->greedy_decode(input, input_mask, EOS_ID, EOS_ID, max_new_toks);
            if (tgt_seq_ids.size(0) != batch_size) {
                throw std::runtime_error("Mismatch in input and output batch sizes");
            }
            for (int i = 0; i < batch_size; ++i) {
                i64* out_seq = tgt_seq_ids[i].cpu().to(torch::kInt64).data_ptr<i64>();
                auto out_seq_vec = vector<int>(out_seq, out_seq + tgt_seq_ids[i].size(0));
                std::cerr << "out_seq_vec: " << out_seq_vec << std::endl;
                string out_seq_str = tgt_vocab->DecodeIds(out_seq_vec);
                std::cout << buffer[i][0] << "\t->\t" << out_seq_str << std::endl;
            }
            };

        vector2d<string> buffer;
        for (auto& line : lines) {
            auto parts = utils::split(line, "\t");
            if (parts.size() < 1) {
                throw std::runtime_error("Invalid input line: " + line + " Expected at least 1 field but got " + std::to_string(parts.size()));
            }
            buffer.push_back({ parts[0] });
            if (buffer.size() >= batch_size) {
                consume_buffer(buffer);
                buffer.clear();
            }
        }
        if (!buffer.empty()) {
            consume_buffer(buffer);
        }
        spdlog::info("Decoding completed");

    }

    void predict(string model_path, vector<string> vocab_paths, string input_file, Pack kwargs) {
        auto [config, model] = utils::restore_model(model_path, DEVICE, /*validate_config=*/false);
        //serialize::store_npz(model_path  + ".restore.npz", model->get_state());

        //std::cerr << "model:\n" << *model << std::endl;
        auto vocabs = utils::load_vocabs(vocab_paths);
        auto data_loader = data::DataLoader(config, vocabs);
        // disable autograd; set model to eval mode
        c10::InferenceMode guard(true);
        model->eval();

        if (model->task_type() == TaskType::REGRESSION) {
            predict_scores(model, data_loader, input_file, kwargs);
        }
        else if (model->task_type() == TaskType::NMT) {
            decode(model, data_loader, input_file, kwargs);
        }
        else {
            throw std::runtime_error("Unsupported task type, only regression is supported for scoring");
        }

    }

} // namespace tahoma::inference