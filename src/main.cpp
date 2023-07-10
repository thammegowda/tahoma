#include <iostream>
#include <argparse.hpp>
#include <torch/torch.h>
#include <toml++/toml.h>
#include <sentencepiece_processor.h>
#include <spdlog/spdlog.h>
#include "nmt/transformer.hpp"

namespace nn = torch::nn;
using namespace std;

namespace rtg {
    auto load_config(const std::string& filename) -> toml::table {
        try {
            toml::table tbl = toml::parse_file(filename);
            return tbl;
        }
        catch (const toml::parse_error& err) {
            std::cerr << "Parsing failed:\n" << err << "\n";
            throw err;
        }
    }

    auto parse_args(int argc, char* argv[]) -> argparse::ArgumentParser {
        argparse::ArgumentParser parser("rtgpp");
        parser.add_argument("-v", "--verbose").help("Increase log verbosity")
            .default_value(false).implicit_value(true);
        parser.add_argument("-c", "--config").help("Config file").required();
        //parser.add_argument("-f", "--foo").help("foo help").default_value(42);
        //parser.add_argument("-b", "--bar").help("bar help");
        //parser.add_argument("-n", "--num").help("num help").default_value(20).scan<'i', int>();
        //parser.add_argument("-r", "--real").help("real help").default_value(20.0).scan<'f', float>();
        //program.add_argument("-m", "--model").help("model name");

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
    spdlog::info("main started");
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [t%t] [%^%l%$] %v");

    auto args = rtg::parse_args(argc, argv);
    spdlog::set_level(args.get<bool>("verbose") ? spdlog::level::debug : spdlog::level::info);

    toml::table config = rtg::load_config(args.get<std::string>("config"));
    std::cerr << config << "\n";


    std::cout << config["model"] << "\n";

    if (!config["model"]) {
        spdlog::error("[model] config block not found");
        return 1;
    }

    // check if model_dim, nhead, num_encoder_layers, num_decoder_layers are present
    auto required_keys = {"model_dim", "attn_head", "encoder_layers", "decoder_layers", "dropout", "ffn_dim"};
    for (auto& key : required_keys) {
        if (!config["model"].as_table()->contains(key)) {
            spdlog::error("model config: {} not found", key);
            return 1;
        }
    }
    // value<T>() returns optional<T>, and then value() returns T
    auto d_model = config["model"]["model_dim"].value_or<int64_t>(-1);
    auto nhead = config["model"]["attn_head"].value_or<int64_t>(-1);
    auto num_encoder_layers = config["model"]["encoder_layers"].value_or<int64_t>(6);
    auto num_decoder_layers = config["model"]["decoder_layers"].value_or<int64_t>(6);
    auto dropout = config["model"]["dropout"].value_or<double>(0.1);
    auto ffn_dim = config["model"]["ffn_dim"].value_or<double>(d_model * 4);

    auto model_config = nn::TransformerOptions(d_model, nhead)
        .num_encoder_layers(num_encoder_layers)
        .dim_feedforward(ffn_dim)
        .dropout(dropout)
        .num_decoder_layers(num_decoder_layers);
    auto model = nn::Transformer(model_config);
    std::cout << model << "\n";

    spdlog::info("main finished");
    return 0;
}