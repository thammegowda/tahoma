#pragma once

#include <iostream>
#include <filesystem>
#include <string>
#include <toml++/toml.h>
#include "commons.hpp"


namespace rtg::config {
    class Config: public toml::table {
    
    public:
        Config(const std::string& filename, bool validate = true)
        : toml::table(toml::parse_file(filename)) {
            if (validate){
                validate_config(*this);
            }
        }

        static auto validate_config(toml::table config) -> void {
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