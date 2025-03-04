#pragma once

#include <iostream>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <cassert>
#include <signal.h>
#include <any>
#include <map>
#include <optional>

#include <backward.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/std.h>
#include "spdlog/sinks/stdout_color_sinks.h"
#include <torch/torch.h>



#define assertm(exp, msg) assert(((void)msg, exp))


namespace tahoma {

    // define short aliases
    using i8 = int8_t;
    using i16 = int16_t;
    using i32 = int32_t;
    using i64 = int64_t;
    //using i128 = long long int;
    //using f16 = half_float::half;
    using f32 = float;
    using f64 = double;
    //using f128 = long double;
    using str = std::string; // DEPRECATED
    using string = std::string;
    using cstr = const char*;

    template<typename T>
    using vector = std::vector<T>;

    template<typename T>
    using vector2d = std::vector<std::vector<T>>;
    namespace fs = std::filesystem;
    namespace torch = torch; // noop, but just incase you forgot
    namespace nn = torch::nn;
    namespace F = torch::nn::functional;
    using namespace torch::indexing;
    using Tensor = torch::Tensor;
    using Slice = torch::indexing::Slice;
    using Device = torch::Device;

    //using Pack = std::map<std::string, std::any>;

    class Pack : public std::map<std::string, std::any> {
    public:

    Pack() = default;
    Pack(std::initializer_list<value_type> init) : std::map<std::string, std::any>(init) {}
    Pack(const Pack& other) = default;
    Pack(Pack&& other) noexcept = default;
    Pack& operator=(const Pack& other) = default;
    Pack& operator=(Pack&& other) noexcept = default;

        template<typename T>
        auto get(const std::string& key, T fallback) -> T const{
            if (this->contains(key)) {
                return std::any_cast<T>(this->at(key));
            } else {
                return fallback;
            }
        }

        template <typename T=Tensor>
        auto get(const std::string& key) -> T {
            return std::any_cast<T>(this->at(key));
        }

    };

    using TensorPack = std::map<std::string, torch::Tensor>;
    //auto k_device = torch::device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    inline auto DEVICE = torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    template<typename T>
    using Ptr = std::shared_ptr<T>;

    /*
    template<typename T>
    using New = std::make_shared<T>;
    template<typename T> using Ptru = std::unique_ptr<T>;
    template<typename T> using Ptrw = std::weak_ptr;
    */

    inline int global_setup() {

        std::locale::global(std::locale(std::getenv("LC_ALL") ? std::getenv("LC_ALL") : "en_US.UTF-8"));
        spdlog::set_default_logger(spdlog::stderr_color_mt("console"));
        spdlog::set_pattern("[%C%m%d %H:%M:%S|t%t][%^%l%$] %v");
        spdlog::set_level(spdlog::level::info);
        spdlog::debug("Global setup started");
        backward::SignalHandling sh;
        return 0;
    }

    enum class Mode {
        TRAINING,
        INFERENCE,
    };


    enum class TaskType {
        LM,
        NMT,
        REGRESSION,
    };
    
    // agggh! hack to get enums cant have to_string method
    inline auto task_type_string(TaskType taskType) -> std::string {
        
        switch (taskType) {
            case TaskType::LM:
                return "LM";
            case TaskType::NMT:
                return "NMT";
            case TaskType::REGRESSION:
                return "REGRESSION";
            default:
                return "UNKNOWN";
        }
    }

} // namespace tahoma
