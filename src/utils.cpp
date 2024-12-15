#include <iostream>
#include <coroutine>
#include <ranges>
#include <memory>

#include <torch/torch.h>
#include <sentencepiece_processor.h>

#include <tahoma.h>
#include <tahoma/model/transformer_nmt.h>
#include <tahoma/model/transformer_lm.h>
#include <tahoma/train/stats_counter.h>
#include <tahoma/train/criterion.h>
#include <tahoma/train/loss_computer.h>
#include <tahoma/utils.h>
#include <tahoma/serialize.h>
#include <tahoma/model/mt5.h>
#include <tahoma/model/metricx.h>


using namespace tahoma;

namespace tahoma::utils {

    auto init_model(config::Config& config, torch::Device& device) -> std::shared_ptr<model::LanguageModel> {
        auto model_type = config["model"]["name"].as<std::string>();
        YAML::Node model_args = config["model"]["args"];
        std::shared_ptr<model::LanguageModel> model;
        if (model_type == "transformer_nmt") {
            model = std::make_shared<model::TransformerNMTImpl>(model_args);
        } else if (model_type == "transformer_lm") {
            model = std::make_shared<model::TransformerLMImpl>(model_args);
        } else if (model_type == "MT5ForRegression") {
            model = std::make_shared<model::metricx::RegressionImpl>(model_args);
        } else if (model_type == "MT5ForConditionalGeneration") {
            model = std::make_shared<model::mt5::ConditionalGenerationImpl>(model_args);
        } else {
            throw std::runtime_error("Unknown model type " + model_type);
        }
        // NOTE: trying to move model to device here causes error. Not sure why.
        //LOG::info("Device: {}", device == torch::kCPU ? "CPU" : "CUDA");
        //model->to(device);
        return model;
    }

    auto restore_model(const std::string& model_path, torch::Device& device, bool validate_config)
        -> std::pair<config::Config, std::shared_ptr<model::LanguageModel>> {
        auto chkpt_state = serialize::load_npz(model_path);
        if (chkpt_state.find("config.yml") == chkpt_state.end()) {
            throw std::runtime_error("config.yml not found in the model file");
        }
        auto config_str = std::any_cast<std::string>(chkpt_state["config.yml"]);
        auto config = config::Config(YAML::Load(config_str), validate_config);
        auto model = init_model(config, device);
        if (chkpt_state.empty()) {
            spdlog::warn("No checkpoint state found, model is initialized with random weights");
        } else {
            spdlog::info("Restoring model state from checkpoint");
            model->set_state(chkpt_state);
        }
        return std::make_pair(config, model);
    }

    auto init_criterion(const YAML::Node& config, i64 ignore_idx) -> nn::AnyModule {
        auto name = config["name"].as<std::string>("cross_entropy");
        if (name == "cross_entropy") {
            f32 label_smooth_rate = config["args"]["label_smooth_rate"].as<f32>(0.0);
            auto criterion = train::CrossEntropyLoss(ignore_idx, label_smooth_rate);
            return nn::AnyModule(criterion);
        } else if (name == "kl_divergence") {
            f32 label_smooth_rate = config["args"]["label_smooth_rate"].as<f32>(0.0);
            i64 num_labels = config["args"]["num_labels"].as<i64>(0);
            if (num_labels < 1) {
                throw std::runtime_error("num_labels must be > 0 for kl_divergence with label_smoothing");
            }
            auto criterion = train::KLDivergence(num_labels, ignore_idx, label_smooth_rate);
            return nn::AnyModule(criterion);
        } else {
            throw std::runtime_error("Unknown criterion " + name + ". only cross_entropy supported");
        }
    }

    //template <typename M>
    auto init_optimizer(const config::Config& config, /*nn::AnyModule*/ std::shared_ptr<model::LanguageModel> model)
        -> std::shared_ptr<optim::Optimizer> {
        auto optim_config = config["optimizer"];
        auto optim_name = optim_config["name"].as<std::string>();
        if (optim_name == "adam") {
            auto options = optim::AdamOptions(optim_config["lr"].as<double>(0.0001));
            if (optim_config["weight_decay"].IsDefined()) {
                options.weight_decay(optim_config["weight_decay"].as<double>());
            }
            if (optim_config["betas"].IsDefined()) {
                auto betas = optim_config["betas"].as<vector<double>>();
                options.betas({ betas[0], betas[1] });
            }
            if (optim_config["eps"].IsDefined()) {
                options.eps(optim_config["eps"].as<double>());
            }
            if (optim_config["amsgrad"].IsDefined()) {
                options.amsgrad(optim_config["amsgrad"].as<bool>());
            }
            spdlog::info("Optimizer {}", optim_name);
            return std::make_shared<optim::Adam>(model->parameters(), options);
        } else {
            throw std::runtime_error("Unknown or unsupported optimizer " + optim_name);
        }
    }

    auto init_scheduler(const config::Config& config, optim::Optimizer& optimizer, i64 initial_step)
        -> std::shared_ptr<train::LRScheduler> {
        i64 start_step = 0; // TODO: restore from checkpt dir tor resume training
        auto scheduler_config = config["scheduler"];
        auto name = scheduler_config["name"].as<std::string>();
        YAML::Node options = scheduler_config["args"];
        if (name == "inverse_sqrt") {
            return std::make_shared<train::InverseSqrtScheduler>(optimizer, start_step, options);
        } else if (name == "noam") {
            return std::make_shared<train::NoamScheduler>(optimizer, start_step, options);
        } else {
            throw std::runtime_error("Unknown or unsupported scheduler " + name);
        }
    }

    auto init_config(fs::path work_dir, fs::path config_file) -> config::Config {
        /*
        * 1. If config_file is not provided, look for config.yaml in work_dir
        * 2. If config_file is provided, copy it to work_dir and use it
        * 3. If config_file is not provided and config.yaml is not found in work_dir, raise error
        */
        auto work_config = work_dir / "config.yaml";
        if (!config_file.empty()) { // given non empty config_file
            if (!fs::is_regular_file(config_file)) {
                throw std::runtime_error(fmt::format("Config file {} not found", config_file.string()));
            }
            if (!fs::exists(work_dir)) {
                spdlog::info("mkdir {}", work_dir);
                fs::create_directories(work_dir);
            }
            spdlog::info("Copy {} ➡️ {}", config_file, work_config);
            fs::copy(config_file, work_config, fs::copy_options::overwrite_existing);
        }
        if (!fs::exists(work_config)) {
            throw std::runtime_error(fmt::format("Config file {} not found", work_config.string()));
        }
        return config::Config(config_file);
    }

    auto load_vocab(const std::string& vocab_path)
        -> std::shared_ptr<sentencepiece::SentencePieceProcessor> {
        auto spp = std::make_shared<sentencepiece::SentencePieceProcessor>();
        if (!fs::exists(vocab_path)) {
            spdlog::error("Vocab file {} not found. Current dir: {}",
                vocab_path, fs::current_path().string());
            throw std::runtime_error("Vocab file " + vocab_path + " not found");
        }
        if (!spp->Load(vocab_path).ok()) {
            throw std::runtime_error("Unable to load vocab from " + vocab_path);
        }
        return spp;
    }

    auto load_vocabs(const std::vector<std::string> vocab_paths)
        -> vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> {
        vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> spps;
        for (auto& vocab_path : vocab_paths) {
            spps.push_back(load_vocab(vocab_path));
        }
        return spps;
    }

    auto subsequent_mask(i64 seq_len, torch::Device device) -> Tensor {
        // input: seq_len
        // pad_idx: padding token id; usually 0; ignore if -1
        // returns: [seq_len, seq_len]
        auto mask = torch::ones({ seq_len, seq_len }, torch::dtype(torch::kInt8).device(device)); // all cells have 1
        mask = torch::triu(mask, /*diagonal=*/1);            // upper triangle and diagonal are 1, lower diagonal are 0
        return mask;
    }

    auto init_loss_computer(const config::Config& config, nn::AnyModule& projector, const i64 pad_id) -> std::shared_ptr<train::LossComputer> {
        auto trainer_criterion = init_criterion(config["trainer"]["criterion"], pad_id);
        std::map<std::string, nn::AnyModule> validation_criteria;
        for (auto criterion_config : config["validator"]["criteria"]) {
            auto name = criterion_config["name"].as<std::string>();
            validation_criteria[name] = init_criterion(criterion_config, pad_id);
        }
        auto chunk_size = config["trainer"]["chunk_size"].as<size_t>(0);
        auto container = std::make_shared<train::CriteriaContainer>(trainer_criterion, validation_criteria);
        return std::make_shared<train::LossComputer>(projector, container, pad_id, chunk_size);
    }

    auto ends_with(const std::string& str, const std::vector<std::string>& candidates) -> bool {
        for (const auto& suffix : candidates) {
            if (str.size() >= suffix.size() && str.rfind(suffix) == str.size() - suffix.size()) {
                return true;
            }
        }
        return false;
    }

    auto read_file(const std::string& path) -> std::string {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file " + path);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    auto read_lines(const std::string& path) -> Generator<std::string> {
        /**
         * Read lines from a file and yield them one by one.
         */
        std::ifstream stream;
        if (path == "-"){
            // FIXME: This wont work on Windows and non POSIX systems
            stream = std::ifstream("/dev/stdin");
        } else {
            stream = std::ifstream(path);
        }
        std::string line;
        while (std::getline(stream, line)) {
            co_yield line;
        }
    }

    auto split(const std::string& text, const std::string& delimiter) -> std::vector<std::string> {
        std::vector<std::string> parts;
        size_t pos = 0;
        size_t start = 0;
        while ((pos = text.find(delimiter, start)) != std::string::npos) {
            parts.push_back(text.substr(start, pos - start));
            start = pos + delimiter.length();
        }
        parts.push_back(text.substr(start));
        return parts;
    }


    auto tensor_repr(Tensor tensor) -> string {

        auto print_indices = [](i64 size, i64 n=6) -> std::vector<i64> {
            std::vector<i64> indices;
            for (auto i = 0; i < std::min(size, n/2); ++i) {
                indices.push_back(i);
            }
            if (size > n) {
                indices.push_back(-1); // Placeholder for "..."
            }
            for (i64 i = std::max(size - n/2, n/2); i < size; ++i) {
                indices.push_back(i);
            }
            return indices;
        };
        auto sizes = tensor.sizes();
        std::ostringstream oss;
        oss << "Tensor with shape [";
        for (size_t i = 0; i < sizes.size(); ++i) {
            oss << sizes[i] << " ";
        }
        oss << "]\n";
        if (tensor.dim() == 0) {
            oss << "Scalar " << tensor.item();
        } else if (tensor.dim() == 1) {
            auto indices = std::min(tensor.size(0), 16L);
            oss << "[";
            for (auto j : print_indices(indices)) {
                if (j == -1) {
                    oss << "...  ";
                    continue;
                }
                oss << tensor[j].item();
                if (j < indices - 1) {
                    oss << ",  ";
                }
            }
            oss << "]";
        } else {
            if (tensor.dim() > 2) {
                oss << "(only the last two dims are shown; 0th item is picked from each higher dim)\n";
                while (tensor.dim() > 2) {
                    tensor = tensor[0];
                }
            }
            oss << "[";
            for (auto i : print_indices(tensor.size(0))) {
                if (i == -1) {
                    oss << "\n  ...\n";
                    continue;
                }
                oss << "\n  [";
                for (auto j : print_indices(tensor.size(1))) {
                    if (j == -1) {
                        oss << "...  ";
                        continue;
                    }
                    oss << tensor[i][j].item();
                    if (j < tensor.size(1) - 1) {
                        oss << ",   ";
                    }
                }
                oss << "]";
            }
            oss << "]";
        }
        return oss.str();
    }

    void debug_message(bool condition, const std::string& message, Tensor data) {
        if (condition) {
            std::ostringstream oss;
            oss << "\n#### " << message << "####";
            if(data.is_floating_point() || data.dtype() == torch::kInt64
                || data.dtype() == torch::kInt32 || data.dtype() == torch::kInt16) {
                oss << " AbsSum: " << data.abs().sum().item<float>()
                    << " Max: " << data.max().item<float>()
                    << " Min: " << data.min().item<float>();
             }
            std::cerr << oss.str() << "\n" << tensor_repr(data) << std::endl;
        }
    }
    void debug_message(bool condition, const std::string& message, Pack& data, std::initializer_list<string> keys) {
        if (!condition) {
            return;
        }
        std::cerr << message << std::endl;
        for (auto key : keys) {
            if (data.find(key) == data.end()) {
                std::cerr << "Key " << key << " not found in data" << std::endl;
                continue;
            }
            debug_message(condition, key, std::any_cast<Tensor>(data[key]));
        }
    }

} // namespace tahoma::train

