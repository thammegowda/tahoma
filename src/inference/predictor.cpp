
#include <tahoma.h>
#include <tahoma/inference/predictor.h>
#include <tahoma/utils.h>
#include <tahoma/vocab.h>
#include <tahoma/data.h>
#include <tahoma/model/metricx.h>
#include <tahoma/serialize.h>

namespace tahoma::inference {

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

    struct OutputCollector {
        std::string out_path;
        std::map<size_t, string> cache;
        size_t idx = 1;

        OutputCollector(string out_path) : out_path(out_path) {}
        ~OutputCollector() {
            if (!cache.empty()) {
                spdlog::error("Cache is not empty; not all scores were sent to output");
            }
        }
        void add(size_t id, string line) {
            // this is thread safe if no two threads outputs the same id, which is the case here
            cache[id] = line;
            while (cache.contains(idx)) {
                std::cout << cache[idx] << std::endl;
                cache.erase(idx);
                idx++;
            }
        }

    };

    struct Predictor {

        config::Config config;
        vector<string> vocab_paths;
        Pack options;  // runtime options from CLI
        std::vector<string> devices;
        std::vector<Ptr<model::LanguageModel>> models;

        data::DataLoader data_loader;
        Predictor(config::Config config, Pack weights, Pack options, std::vector<string> devices): 
        config(config),
        options(options),
        devices(devices),
        data_loader(config, utils::load_vocabs(options.get("vocab_paths", vector<string>{}))),
        models(devices.size())
        {
            std::vector<std::jthread> threads;
            for (auto idx = 0; idx < devices.size(); idx++) {
                auto t = std::jthread([&, idx] {
                    auto dev_name = devices[idx];
                    spdlog::info("Initializing model on device {}",  dev_name);
                    torch::Device device(dev_name);
                    auto model = utils::init_model(config, device);
                    model->set_state(weights);
                    model->eval();
                    model->to(device);
                    models[idx] = model;
                });
                threads.push_back(std::move(t));
            }
            for (auto& t: threads) {
                t.join();
            }
        }
        
        void predict_scores(string input_file, string output_file){

            auto regression_model = std::dynamic_pointer_cast<model::metricx::RegressionImpl>(models[0]);
            auto mloader = data::MultiThreadedLoader(data_loader, {input_file}, 1, 1, false, {1024},
                 {std::make_shared<MetricxLineMapper>(regression_model, options.get("qe", false))});
            mloader.add_eos = false;
            size_t n_data_threads = options.get("data_threads", std::max((size_t)4, devices.size()));
            mloader.start(n_data_threads);

            auto collector = OutputCollector(output_file); 
            size_t batch_count = 0;
            
            std::queue<data::Batch> queue;
            bool done = false;
            size_t max_queue_size = 16 * devices.size();
            std::mutex queue_mutex;
            std::condition_variable queue_cv;

            // load to queue
            auto dispatcher = std::jthread([&] {
                for (auto batch: mloader.generator()){
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&] { return queue.size() < max_queue_size; });
                    queue.push(batch);
                    lock.unlock();
                    queue_cv.notify_all();
                }
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    done = true;
                }
            });

            std::vector<std::jthread> threads;
            for (size_t idx = 0; idx < devices.size(); ++idx) {
                thread_local auto pad_id = data_loader.vocabs[0]->pad_id();
                auto t = std::jthread(
                    [&, idx] {
                    auto dev_name = this->devices[idx];
                    spdlog::info("Starting thread on device {}", dev_name);
                    thread_local auto device = torch::Device(dev_name);
                    thread_local auto model = this->models[idx];

                    while (true) {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        queue_cv.wait(lock, [&] { return done || !queue.empty(); });
                        if (queue.empty()) {
                            if (done) {
                                break;
                            }
                            continue;
                        }

                        auto batch = queue.front();
                        queue.pop();
                        lock.unlock();
                        queue_cv.notify_all();

                        batch = batch.contiguous().to(device);
                        auto inps = batch.fields[0];
                        auto mask = inps.eq(pad_id).to(torch::kBool).view({ inps.size(0), 1, 1, inps.size(1) });
                        Pack inps_pack = {
                            {"input", inps},
                            {"input_mask", mask},
                        };
                        auto out = model->forward(inps_pack);
                        auto scores = std::any_cast<torch::Tensor>(out["result"]);
                        for (int i = 0; i < scores.size(0); ++i) {
                            string line = fmt::format("{:.6f}", scores[i].item<float>());
                            collector.add(batch.examples[i].id, line);
                        }
                    }
                });
                threads.push_back(std::move(t));
            }
            spdlog::info("Waiting for threads to finish");
            dispatcher.join();
            for (auto& t: threads) {
                t.join();
            }
            spdlog::info("All threads finished");
        }
        
    };



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

    void predict(string model_path, string input_file, Pack kwargs) {
        auto [config, weights] = utils::load_checkpt(model_path, /*validate_config=*/false);
        auto model_name = config["model"]["name"].as<string>();
        std::vector<string> devices;
        if (torch::cuda::is_available()) {
            for (auto i = 0; i < torch::cuda::device_count(); i++) {
                devices.push_back(torch::Device(torch::kCUDA, i).str());
            }
        } else {
            devices.push_back(torch::Device(torch::kCPU).str());
        }
        spdlog::info("Devices: {}", fmt::join(devices, ", "));
        if (model_name == "MT5ForRegression") {
            auto predictor = Predictor(config, weights, kwargs, devices);
            predictor.predict_scores(input_file, "-");
            return;
        } else {
            std::cerr << "Unsupported model type: " << model_name << std::endl;
        }

    }

} // namespace tahoma::inference