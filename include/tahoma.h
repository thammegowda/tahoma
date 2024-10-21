#pragma once
//#define BACKWARD_HAS_UNWIND 1

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

#include "__generator.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/format.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>
#include <sentencepiece_processor.h>


#define assertm(exp, msg) assert(((void)msg, exp))

// define macro for aliasing std::shared_ptr<T> as Ptr<T>
//#define Ptr std::shared_ptr
//#define New std::make_shared


/// @brief formatter for filesystem:path to work with spdlog
// https://stackoverflow.com/a/69496952/1506477

template<>
struct fmt::formatter<std::filesystem::path> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const std::filesystem::path& input, FormatContext& ctx) -> decltype(ctx.out()) {
        return format_to(ctx.out(), "{}", std::string(input));
    }
};


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
    using str = std::string;
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
    using Pack = std::map<std::string, std::any>;
    //auto k_device = torch::device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    /*
    template<typename T>
    using ptr = std::shared_ptr<T>;

    template<typename T>
    using u_ptr = std::unique_ptr<T>;

    template<typename T>
    using w_ptr = std::weak_ptr;
    */
    inline int global_setup() {
        spdlog::info("Global setup started");
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [t%t] [%^%l%$] %v");
        spdlog::set_level(spdlog::level::info);
        return 0;
    }

    enum class Mode {
        TRAINING,
        INFERENCE,
    };


    enum class TaskType {
        LM,
        NMT,
    };


} // namespace tahoma
