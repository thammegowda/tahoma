#pragma once
#include <yaml-cpp/yaml.h>
#include <tahoma.h>

namespace tahoma::config {

    class Config: public YAML::Node {

    public:
        Config(const YAML::Node& other, bool validate=true ) : YAML::Node(other) {
            if (validate) {
                validate_config(*this);
            }
        }
        Config(const std::string& filename, bool validate = true) :
            Config(YAML::LoadFile(filename), validate) {}

        Config(const Config& other) = default;
        Config(Config&& other) = default;
        Config& operator=(const Config& other) = default;
        Config& operator=(Config&& other) = default;
        ~Config() = default;
        static void validate_config(const YAML::Node& config);
    };

}