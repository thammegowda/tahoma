#include <iostream>
#include <argparse.hpp>
#include <tahoma.h>
#include <tahoma/train/trainer.h>
#include <tahoma/inference/predictor.h>

using namespace tahoma;

std::string get_version() {
    // TODO: read from file
    return "0.0.1";
}


int main(int argc, char* argv[]) {
    if (global_setup() != 0) {
        return 1;
    }
    spdlog::info("main started.. torch version: {} ", TORCH_VERSION);
    auto parser = argparse::ArgumentParser("tahoma", get_version());

    argparse::ArgumentParser train_cmd("train", "", argparse::default_arguments::help);
    train_cmd.add_argument("work_dir")
        .help("Working directory")
        .required();
    train_cmd.add_argument("-c", "--config")
        .help("Config file. Optional: default is config.toml in work_dir");

    argparse::ArgumentParser predict_cmd("predict", "", argparse::default_arguments::help);
    predict_cmd.add_argument("-m", "--model")
        .help("Path to model")
        .required();
    predict_cmd.add_argument("-v", "--vocabs")
        .help("Paths to vocabulary files")
        .nargs(argparse::nargs_pattern::at_least_one)
        .required();
    predict_cmd.add_argument("-i", "--input")
        .help("Input file containing TSV records. '-' for stdin. For metrics, the following field order is expected: source, hypothesis, reference")
        .default_value("-");

    predict_cmd.add_argument("-bs", "--batch-size")
        .scan<'d', int>()
        .help("Batch Size")
        .default_value(1);

    predict_cmd.add_argument("-qe", "--qe")
        .help("Quality Estimation model")
        .default_value(false)
        .implicit_value(true);
    predict_cmd.add_argument("-pm", "--print-model")
        .help("Print model architecture")
        .default_value(false)
        .implicit_value(true);

    parser.add_argument("-V", "--verbose")
        .help("Increase log verbosity")
        .default_value(false)
        .implicit_value(true);
    parser.add_subparser(train_cmd);
    parser.add_subparser(predict_cmd);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }

    /// =====================
    if (parser.get<bool>("verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }
    if (torch::cuda::is_available()) {
        std::vector<std::string> device_ids;
        for (auto i = 0; i < torch::cuda::device_count(); i++) {
            device_ids.push_back(fmt::format("{}", i));
        }
        spdlog::info("CUDA devices: {}", fmt::join(device_ids, ", "));
    } else {
        spdlog::info("CUDA is not available");
    }

    if (parser.is_subcommand_used("train")) {
        auto work_dir = fs::path{ train_cmd.get<std::string>("work_dir") };
        auto config_file_arg = train_cmd.get<std::string>("config");
        fs::path config_file;
        if (!config_file_arg.empty()) {
            config_file = fs::path{ config_file_arg };
        }
        auto trainer = train::Trainer(work_dir, config_file);
        trainer.train();
    } else if (parser.is_subcommand_used("predict")) {
        auto model_path = predict_cmd.get<std::string>("model");
        auto vocab_paths = predict_cmd.get<std::vector<std::string>>("vocabs");
        auto input_file = predict_cmd.get<std::string>("input");
        Pack pred_args = {
            {"qe", predict_cmd.get<bool>("qe")},
            {"print_model", predict_cmd.get<bool>("print-model")},
            {"batch_size", predict_cmd.get<int>("batch-size")},
            {"vocab_paths", vocab_paths},
        };
        inference::predict(model_path, input_file, pred_args);
    } else {
        std::cerr << "Unknown command. Knwon commands: train, predict\n";
        std::cerr << parser;
        return 2;
    }
    spdlog::info("main finished..");
    return 0;
}