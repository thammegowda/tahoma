
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
        size_t max_length = kwargs.get("max_length", 1024);
        size_t data_threads = kwargs.get("data_threads", 4);


        assertm(model->task_type() == TaskType::REGRESSION, "Only regression models are supported for scoring");
        // TODO: support other models
        auto regression_model = std::dynamic_pointer_cast<model::metricx::RegressionImpl>(model);
        if (regression_model == nullptr) {
            throw std::runtime_error("Failed to cast model to RegressionImpl");
        }

        auto PAD_ID = loader.vocabs[0]->pad_id();
        torch::Device device = DEVICE;
        vector<i32> eos_ids = {}; // EOS IDs are not required. Google's code removes it from HF tokenizer's output
        vector<size_t> max_lens = { 1024 };

        regression_model->eval();
        regression_model->to(device);

        class MetricxLineMapper: public data::LineMapper {
            tahoma::Ptr<model::metricx::RegressionImpl> regression_model;
            bool is_qe;

            public:
            MetricxLineMapper(tahoma::Ptr<model::metricx::RegressionImpl> regression_model, bool is_qe)
                : regression_model(regression_model), is_qe(is_qe) {}
            
            auto map(const std::string& line) -> std::string override {
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
                auto input_field = regression_model->make_input(example, is_qe);
                return input_field;
            }
        };


        std::vector<size_t> max_lengths = { max_length };
        std::vector<Ptr<data::LineMapper>> input_mappers;
        input_mappers.push_back(std::make_shared<MetricxLineMapper>(regression_model, is_qe));

        auto mloader = data::MultiThreadedLoader(loader, {file_name}, batch_size, /*maxi_batch=*/1, /*max_length_crop=*/false, max_lengths, input_mappers);
        mloader.add_eos = false; // EOS IDs should not be added. Google's code removes it from HF tokenizer's output
        std::map<size_t, string> cache;

        mloader.start(data_threads);
        size_t idx = 1;
        for (auto batch: mloader.generator()){
            batch = batch.contiguous().to(device);
            auto inps = batch.fields[0];
            auto mask = inps.eq(PAD_ID).to(torch::kBool).to(device).view({ inps.size(0), 1, 1, inps.size(1) });
            Pack inps_pack = {
                {"input", inps},
                {"input_mask", mask},
            };
            auto out = regression_model->forward(inps_pack);
            auto scores = std::any_cast<torch::Tensor>(out["result"]);
            for (int i = 0; i < scores.size(0); ++i) {
                cache[batch.examples[i].id] = fmt::format("{:.{}f}", scores[i].item<float>(), width); // cache the score
            }
            while (cache.contains(idx)) {
                std::cout << cache[idx] << std::endl; // << "\t" << idx  
                cache.erase(idx);
                idx++;
            }
        }
        if (!cache.empty()) {
            throw std::runtime_error("Cache is not empty; not all scores were printed");
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

        auto consume_buffer = [&](vector<data::IdRawExample> buffer) -> void {
            vector<data::Example> examples;
            for (auto& [id, fields] : buffer) {
                auto example = loader.make_example(id, fields, eos_ids, max_lens, /*max_length_crop=*/true);
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
                string src_str = buffer[i].second[0];
                string out_str = tgt_vocab->DecodeIds(out_seq_vec);
                std::cout << src_str << "\t->\t" << out_str << std::endl;
            }
            };

        vector<data::IdRawExample> buffer;
        size_t rec_num = 0;
        for (auto& line : lines) {
            auto parts = utils::split(line, "\t");
            if (parts.size() < 1) {
                throw std::runtime_error("Invalid input line: " + line + " Expected at least 1 field but got " + std::to_string(parts.size()));
            }
            buffer.push_back({++rec_num, { parts[0] }});
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
        bool print_model = kwargs.get("print_model", false);
        if (print_model) {
            std::cerr << "model:\n" << *model << std::endl;
        }

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