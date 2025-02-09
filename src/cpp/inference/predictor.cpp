
#include <tahoma.h>
#include <tahoma/inference/predictor.h>
#include <tahoma/utils.h>
#include <tahoma/vocab.h>
#include <tahoma/data.h>
#include <tahoma/model/metricx.h>
#include <tahoma/serialize.h>
#include <tahoma/autocast.h>

namespace tahoma::inference {

    class MetricxLineMapper : public data::LineMapper {
        tahoma::Ptr<model::metricx::RegressionImpl> regression_model;
        bool is_qe;

    public:
        MetricxLineMapper(tahoma::Ptr<model::metricx::RegressionImpl> regression_model, bool is_qe)
            : regression_model(regression_model), is_qe(is_qe) {
        }

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
        std::mutex mutex;
        std::condition_variable cv;

        OutputCollector(string out_path) : out_path(out_path) {}
        ~OutputCollector() {
            if (!cache.empty()) {
                // @TG: this should not be happening. There was a bug in the earlier version of code and I believe I have fixed it.
                //   we should  remove this after some testing
                spdlog::error("Cache is not empty; not all scores were sent to output. Remaining items in cache {}", cache.size());
                spdlog::error("Awating: {}, Dumping cache to stderr", idx);
                for (auto& [id, line] : cache) {
                    std::cerr << "cached:" << id << "\t" << line << std::endl;
                }
            }
        }
        void put(const size_t id, const string& line) {
            // this is thread safe if no two threads outputs the same id, which is the case here
            if (cache.contains(id)) {
                throw std::runtime_error("Duplicate id found in cache: " + std::to_string(id));
            }
            std::lock_guard<std::mutex> lock(mutex);
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
        Pack options;  // runtime options from CLI
        std::vector<Device> devices;
        vector<string> vocab_paths;
        data::DataLoader data_loader;
        std::vector<Ptr<model::LanguageModel>> models;
        bool use_fp16 = false;

        Predictor(config::Config config, Pack weights, Pack options, std::vector<Device> devices) :
            config(config),
            options(options),
            devices(devices),
            data_loader(config, utils::load_vocabs(options.get("vocab_paths", vector<string>{}))),
            models(devices.size()),
            use_fp16(options.get("fp16", false)) {
            auto start_time = std::chrono::system_clock::now();
            std::vector<std::jthread> threads;
            for (auto idx = 0; idx < devices.size(); idx++) {
                auto t = std::jthread([&, idx] {
                    auto device = devices[idx];
                    spdlog::info("[{} {}] WIP; Initializing model", idx, device.str());
                    auto device_guard = torch::DeviceGuard(device);
                    auto model = utils::init_model(config, device);
                    spdlog::info("[{} {}] DONE; Initialized model", idx, device.str());
                    model->set_state(weights);
                    model->to(device);
                    models[idx] = model;
                    });
                threads.push_back(std::move(t));
            }
            for (auto& t : threads) {
                t.join();
             }
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start_time);
            spdlog::info("Predictor initialized with {} workers... elapsed: {:L}s", devices.size(), elapsed.count());
        }

        void predict_scores(string input_file, string output_file) {

            auto regression_model = std::dynamic_pointer_cast<model::metricx::RegressionImpl>(models[0]);
            size_t mini_batch = options.get<size_t>("mini_batch", 1);
            size_t maxi_batch = options.get<size_t>("maxi_batch", 1);
            size_t n_data_threads = options.get("data_threads", std::max((size_t)4, devices.size()));
            auto line_mapper = std::make_shared<MetricxLineMapper>(regression_model, options.get("qe", false));
            size_t max_length = options.get<size_t>("max_length", regression_model->max_input_length());
            bool max_length_crop = true;
            auto collector = OutputCollector(output_file);
            auto mloader = data::MultiThreadedLoader(
                data_loader, { input_file }, mini_batch, maxi_batch,
                max_length_crop, { max_length }, { line_mapper });
            mloader.add_eos = false;
            mloader.sort_by = "length";
            mloader.start(n_data_threads);

            auto pad_id = data_loader.vocabs[0]->pad_id();
            std::vector<std::jthread> threads;
            for (size_t worker_id = 0; worker_id < devices.size(); ++worker_id) {
                auto t = std::jthread([&, worker_id, pad_id] {
                    auto device = devices[worker_id];
                    spdlog::info("Starting predict task on {}", device.str());
                    auto autocast_guard = AutoCastGuard(device.type(), use_fp16 && torch::autocast::is_autocast_enabled(device.type()));
                    if (autocast_guard.is_enabled() && worker_id == 0) {
                        spdlog::info("Enabled FP16");
                    }
                    auto device_guard = torch::DeviceGuard(device);
                    auto inference_guard = torch::InferenceMode(); // has to be on each thread
                    auto& model = this->models[worker_id];
                    model->eval();
                    auto& queue = mloader.mini_batch_queue;
                    size_t count = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mloader.mutex);
                        mloader.cv.wait(lock, [&] {return !queue.empty() || mloader.status.is_done(); });
                        if (mloader.status.is_done() && queue.empty()) {
                            break;
                        }
                        auto batch = queue.front();
                        queue.pop();
                        lock.unlock();
                        mloader.cv.notify_one();

                        batch = batch.contiguous().to(device);
                        auto inps = batch.fields[0];
                        auto mask = inps.eq(pad_id).to(torch::kBool).view({ inps.size(0), 1, 1, inps.size(1) });
                        Pack inps_pack = {
                            {"input", inps},
                            {"input_mask", mask},
                        };
                        Pack out;
                        try {
                            out = model->forward(inps_pack);
                        } catch(const c10::OutOfMemoryError& e) {
                            auto err_msg = fmt::format("OOM error on worker {}. input_size: {}; Inner exception what: {} ", worker_id, fmt::join(inps.sizes(), ", "), e.what());
                            throw std::runtime_error(err_msg);
                        }
                        auto scores = std::any_cast<torch::Tensor>(out["result"]);
                        for (int i = 0; i < scores.size(0); ++i) {
                            auto ex = batch.examples[i];
                            //string line = fmt::format("{:.6f}\t{}\tlength: {}", scores[i].item<float>(), ex.id, ex.fields[0].size());
                            string line = fmt::format("{:.6f}", scores[i].item<float>());
                            collector.put(ex.id, line);
                            count++;
                        }
                    }
                    spdlog::info("Worker {} finished; total items processed: {}", worker_id, count);
                    });
                threads.push_back(std::move(t));
            }
            spdlog::debug("Waiting for all worker threads to finish");
            for (auto& t : threads) {
                t.join();
            }
            spdlog::debug("All threads finished");
        }
    };

    void predict(string model_path, string input_file, Pack kwargs) {
        auto [config, weights] = utils::load_checkpt(model_path, /*validate_config=*/false);
        auto model_name = config["model"]["name"].as<string>();
        std::vector<Device> devices;
        if (torch::cuda::is_available()) {
            for (auto i = 0; i < torch::cuda::device_count(); i++) {
                devices.push_back(torch::Device(torch::kCUDA, i));
            }
        } else {
            devices.push_back(torch::Device(torch::kCPU));
        }
        if (model_name == "MT5ForRegression") {
            auto predictor = Predictor(config, weights, kwargs, std::move(devices));
            predictor.predict_scores(input_file, "-");
            return;
        } else {
            std::cerr << "Unsupported model type: " << model_name << std::endl;
        }
    }

} // namespace tahoma::inference