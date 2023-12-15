#pragma once

#include <fstream>
#include <filesystem>
#include <__generator.hpp>
#include <random>
#include <algorithm>
#include <vector>
#include <string>


namespace rtg::utils {

    std::generator<std::string> read_lines(std::string path){
        /**
         * Read lines from a file and yield them one by one.
        */
        std::ifstream file(path);
        std::string line;
        while(std::getline(file, line)){
            co_yield line;
        }
        file.close();
    }

    template <typename T>
    auto sample_n_items(const std::vector<T>& buffer, i32 n) -> std::vector<T>{
        std::vector<T> samples = buffer; // copy the original vector
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(samples.begin(), samples.end(), g);
        // If n is greater than the size of the vector, return the whole vector
        if (n > samples.size()) {
            return samples;
        }
        samples.resize(n); // resize the vector to contain only the first n elements
        return samples;
    }

    template <typename T>
    auto sample_n_items(std::generator<T> stream, i32 n) -> std::generator<T> {
        // buffer -> sample -> yield
        vector<vector<string>> buffer;
        for (auto item : stream) {
            buffer.push_back(item);
        }
        auto samples = sample_n_items(buffer, n);
        for (auto sample : samples) {
            co_yield sample;
        }
    }

    auto tensor_shape(Tensor tensor) -> string {
        /**
         * Return the shape of a tensor as a string.
        */
        string shape = "";
        for (auto size : tensor.sizes()) {
            if (shape != ""){
                shape += ", ";
            }
            shape += std::to_string(size);
        }
        return "[" + shape + "]";

    }
}