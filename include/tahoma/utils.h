
#pragma once
#include <tahoma.h>

namespace tahoma::utils {

    std::generator<std::string> read_lines(std::string path);

    template <typename T>
    auto sample_n_items(const std::vector<T>& buffer, i32 n) -> std::vector<T>;

    template <typename T>
    auto sample_n_items(std::generator<T> stream, i32 n) -> std::generator<T>;

    auto tensor_shape(Tensor tensor) -> std::string;
}