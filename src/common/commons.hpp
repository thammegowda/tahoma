#pragma once
//#define BACKWARD_HAS_UNWIND 1

#include <iostream>
#include <filesystem>
#include <memory>
#include <cassert>
#include <signal.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/format.h>

namespace fs = std::filesystem;
namespace LOG = spdlog;

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

//template<typename T>
//using UPtr<T> = std::unique_ptr<T>;
//template<typename T>
//using Ptr = std::shared_ptr<T>;
// using wptr = std::weak_ptr;

#define assertm(exp, msg) assert(((void)msg, exp))


/// @brief formatter for filesystem:path to work with spdlog
// https://stackoverflow.com/a/69496952/1506477
template<>
struct fmt::formatter<fs::path> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const fs::path& input, FormatContext& ctx) -> decltype(ctx.out()) {
        return format_to(ctx.out(), "{}", std::string(input));
    }
};
///

int global_setup() {
    LOG::info("Global setup started");
    LOG::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [t%t] [%^%l%$] %v");
    LOG::set_level(LOG::level::info);
    return 0;
}
