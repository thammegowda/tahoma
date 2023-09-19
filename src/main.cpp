#include <iostream>
#include <filesystem>
#include <argparse.hpp>
#include <toml++/toml.h>
#include <sentencepiece_processor.h>

#include <torch/torch.h>
#include "common/commons.hpp"
#include "common/config.hpp"
#include "nmt/transformer.hpp"
#include "nmt/trainer.hpp"



namespace nn = torch::nn;
namespace optim = torch::optim;
namespace fs = std::filesystem;


namespace rtg {
    
    auto parse_args(int argc, char* argv[]) -> argparse::ArgumentParser {
        argparse::ArgumentParser parser("rtgp");
        parser.add_argument("-v", "--verbose").help("Increase log verbosity")
            .default_value(false).implicit_value(true);
        parser.add_argument("work_dir").help("Working directory").required();
        parser.add_argument("-c", "--config").help("Config file. Optional: default is config.toml in work_dir");

        try {
            parser.parse_args(argc, argv);
        }
        catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::cerr << parser;
            exit(1);
        }
        return parser;
    }

   

} // namespace rtg


int main(int argc, char* argv[]) {

    int _code = global_setup();
    if (_code != 0){
        return _code;
    }

    auto args = rtg::parse_args(argc, argv);
    if (args.get<bool>("verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }

    auto work_dir = fs::path {args.get<std::string>("work_dir")};
    spdlog::info("work_dir: {}", work_dir);
    auto config_file_arg = args.get<std::string>("config");
    fs::path config_file =  work_dir / "config.toml";
    if (!fs::exists(work_dir)){
        spdlog::info("Creating work dir {}", work_dir);
        fs::create_directories(work_dir);
    }
    if (!config_file_arg.empty()){
        spdlog::info("copying config file {} -> {}", config_file_arg, config_file);
        fs::copy(fs::path(config_file_arg), config_file,
                    fs::copy_options::overwrite_existing);
    }
    if (!fs::exists(config_file)) {
        spdlog::error("config file {} not found", config_file);
        throw std::runtime_error("config file" + std::string(config_file) + "not found");
    }

    auto config = rtg::config::Config(config_file);
    auto model = rtg::nmt::transformer::init_model(config);
    auto criterion = nn::CrossEntropyLoss();
    auto optimizer = optim::Adam(model.ptr() -> parameters(), optim::AdamOptions(0.0001));
    auto scheduler = optim::StepLR(optimizer, 1.0, 0.95);


    //std::vector<std::string> data_paths { "data/train.src", "data/train.tgt" };
    //std::vector<std::string> vocab_paths { "data/vocab.src", "data/vocab.tgt" };
    /* lambda to comvert toml array to vector of strings */
    auto as_string_vector = [](const toml::array& arr) -> std::vector<std::string> {
        std::vector<std::string> vec;
        for (const auto& v : arr) {vec.push_back(v.as_string()->get());}
        return vec;
    };
    auto data_paths = as_string_vector(*config["trainer"]["data"].as_array());
    auto vocab_paths =  as_string_vector(*config["schema"]["vocabs"].as_array());

    rtg::trainer::TrainerOptions options {
        .data_paths = data_paths,
        .vocab_paths = vocab_paths,
        .epochs = 10,
        .batch_size = 32
    };

    auto trainer = rtg::trainer::Trainer(nn::AnyModule(model),
                     optimizer, scheduler, nn::AnyModule(criterion), options);
    trainer.train(options);
    spdlog::info("main finished..");
    return 0;
}