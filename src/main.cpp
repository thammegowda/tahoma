#include <iostream>
#include <argparse.hpp>
#include <torch/torch.h>
#include "../libs/sentencepiece/src/sentencepiece_processor.h"


int main(int argc, char* argv[]) {

    argparse::ArgumentParser parser("program_name");
    parser.add_argument("--verbose")
        .help("increase output verbosity")
        .default_value(false)
        .implicit_value(true);
    // dummy
    parser.add_argument("-f", "--foo").help("foo help").default_value(42);
    parser.add_argument("-b", "--bar").help("bar help");
    parser.add_argument("-n", "--num").help("num help").default_value(20).scan<'i', int>();
    parser.add_argument("-r", "--real").help("real help").default_value(20.0).scan<'f', float>();

    //program.add_argument("-m", "--model").help("model name");

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        exit(0);
    }

    torch::Tensor tensor = torch::rand({ 2, 3 });
    std::cout << tensor << std::endl;
    return 0;
}