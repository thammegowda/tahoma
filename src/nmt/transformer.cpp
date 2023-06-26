#include <iostream>
#include "torch/torch.h"

namespace nn = torch::nn;

int main(int argc, char* argv[]) {
    //auto config = nn::TransformerOptions({.d_model=512, .nhead=8, .num_encoder_layers=6, .num_decoder_layers=6});  // C++20
    auto config = nn::TransformerOptions(/*dim=*/512, /*head=*/8, /*encoder_layers=*/6, /*decoder_layers=*/6);
    auto model = nn::Transformer(config);
    std::cout << model << '\n';
    return 0;
}