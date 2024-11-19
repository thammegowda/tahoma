#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <type_traits>
#include <yaml-cpp/yaml.h>
#include <tahoma.h>
#include <tahoma/config.h>

namespace tahoma::config {


    Config::Config(const std::string& filename, bool validate)
        : YAML::Node {YAML::LoadFile(filename)} {
            if (validate){
                validate_config(*this);
            }
    }

    void Config::validate_config(const YAML::Node& config) {
        std::vector<std::string> expected_keys = { "model", "schema", "optimizer", "trainer", "validator" };
        for (auto key : expected_keys) {
            if (!config[key]) {
                spdlog::error("config key {} not found", key);
                throw std::runtime_error("config key " + key + " not found");
            }
        }
    }


} // namespace tahoma::config