#include <iostream>
#include <argparse.hpp>
#include <tahoma.h>
#include <tahoma/train/trainer.h>

using namespace tahoma;

 auto trainer_args(int argc, char* argv[]) -> argparse::ArgumentParser {
    argparse::ArgumentParser parser("trainer");
    parser.add_argument("-V", "--verbose").help("Increase log verbosity")
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

int trainer_main(int argc, char* argv[]){
    spdlog::info("main started.. torch version: {} ", TORCH_VERSION);
    auto args = trainer_args(argc, argv);
    if (args.get<bool>("verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }
    if (torch::cuda::is_available()) {
        std::vector<std::string> device_ids;
        for (auto i=0; i < torch::cuda::device_count(); i++){
            device_ids.push_back(fmt::format("{}", i));
        }
        spdlog::info("CUDA devices: {}", fmt::join(device_ids, ", "));
    } else {
        spdlog::info("CUDA is not available");
    }

    auto work_dir = fs::path{ args.get<std::string>("work_dir") };
    auto config_file_arg = args.get<std::string>("config");
    fs::path config_file;
    if (!config_file_arg.empty()) {
        config_file = fs::path{ config_file_arg };
    }
    auto trainer = train::Trainer(work_dir, config_file);
    trainer.train();
    spdlog::info("main finished..");
    return 0;
}


int main(int argc, char* argv[]) {
    int _code = global_setup();
    if (_code != 0){
        return _code;
    }
    return trainer_main(argc, argv);
}