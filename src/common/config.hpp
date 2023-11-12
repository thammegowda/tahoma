#pragma once

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <type_traits>
#include <yaml-cpp/yaml.h>
#include <rtg.hpp>


namespace rtg::config {

    class Config: public YAML::Node {
    
    public:
        Config(const std::string& filename, bool validate = true)
        : YAML::Node {YAML::LoadFile(filename)} {
            if (validate){
                validate_config(*this);
            }
        }
        // copy and move
        Config(const Config& other) = default;
        Config(Config&& other) = default;
        Config& operator=(const Config& other) = default;
        Config& operator=(Config&& other) = default;
        // destructor
        ~Config() = default;
        
        static auto validate_config(const YAML::Node& config) -> void {
            std::vector<std::string> expected_keys = { "model", "schema", "optimizer", "trainer", "validator" };
            for (auto key : expected_keys) {
                if (!config[key]) {
                    spdlog::error("config key {} not found", key);
                    throw std::runtime_error("config key " + key + " not found");
                }
            }
        }

    };
    
} // namespace rtg::config