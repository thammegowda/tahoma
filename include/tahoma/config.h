#pragma once
#include <yaml-cpp/yaml.h>
#include <tahoma.h>

namespace tahoma::config {

    class Config: public YAML::Node {

    public:
        Config(const std::string& filename, bool validate = true);
        Config(const Config& other) = default;
        Config(Config&& other) = default;
        Config& operator=(const Config& other) = default;
        Config& operator=(Config&& other) = default;
        ~Config() = default;
        static void validate_config(const YAML::Node& config);
    };

}