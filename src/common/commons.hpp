#pragma once
//#define BACKWARD_HAS_UNWIND 1

#include <iostream>
#include <filesystem>
#include <signal.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/format.h>

namespace fs = std::filesystem;
namespace LOG = spdlog;

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
