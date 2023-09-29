#pragma once

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <type_traits>
//#include <toml++/toml.h>
#include <yaml-cpp/yaml.h>
#include "commons.hpp"


namespace rtg::config {
    class Config: public YAML::Node {
    
    public:
        Config(const std::string& filename, bool validate = true)
        : YAML::Node {YAML::LoadFile(filename)} {
            if (validate){
                validate_config(*this);
            }
        }

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