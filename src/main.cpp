#include <iostream>
#include <filesystem>
#include <argparse.hpp>
#include <toml++/toml.h>
#include <sentencepiece_processor.h>

#include <torch/torch.h>
#include "common/commons.hpp"
#include "nmt/transformer.hpp"
#include "nmt/trainer.hpp"


namespace nn = torch::nn;
namespace optim = torch::optim;
namespace fs = std::filesystem;


namespace rtg {
    auto load_config(const std::string& filename) -> toml::table {
        try {
            toml::table tbl = toml::parse_file(filename);
            std::cerr << tbl << "\n";
            return tbl;
        }
        catch (const toml::parse_error& err) {
            std::cerr << "Parsing failed:\n" << err << "\n";
            throw err;
        }
    }

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

    auto init_data(toml::table config) -> void {
        if (!config["data"]) {
            spdlog::error("[data] config block not found");
            throw std::runtime_error("[data] config block not found");
        }
        auto data_conf = config["data"];
        std::cout << data_conf << "\n";
        std::cout << data_conf["train"] << "\n";
        std::cout << data_conf["validation"] << "\n";
        std::cout << data_conf["vocabulary"] << "\n";
        std::cout << data_conf["max_length"] << "\n";
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

    toml::table config = rtg::load_config(config_file);
    rtg::init_data(config);
    auto model = rtg::nmt::transformer::init_model(config);
    auto criterion = nn::CrossEntropyLoss();
    auto optimizer = optim::Adam(model.ptr() -> parameters(), optim::AdamOptions(0.0001));
    auto scheduler = optim::StepLR(optimizer, 1.0, 0.95);

    auto trainer = rtg::trainer::Trainer(nn::AnyModule(model), optimizer, scheduler, nn::AnyModule(criterion));
    rtg::trainer::TrainerOptions options {
        .data_paths = { "data/train.src", "data/train.tgt" },
        .vocab_paths = { "data/vocab.src", "data/vocab.tgt" },
        .epochs = 10,
        .batch_size = 32
    };
    trainer.train(options);
    spdlog::info("main finished..");
    return 0;
}