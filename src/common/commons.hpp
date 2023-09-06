#define BACKWARD_HAS_UNWIND 1

#include <iostream>
#include <filesystem>
#include <signal.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/format.h>
//#include <stacktrace>
#include <backward.hpp>

namespace fs = std::filesystem;

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

void signal_handler(int sig) {

    spdlog::error("signal {}", sig);
    //std::cerr << std::stacktrace::current();

    using namespace backward;
    StackTrace st;
    st.load_here(32);
    st.skip_n_firsts(3);

    Printer p;
    p.object = true;
    p.color_mode = ColorMode::always;
    p.address = true;
    p.snippet = true;
    p.trace_context_size = 3;
    p.print(st, stderr);

    // exit with signal number
    exit(sig);
}



int global_setup() {
    spdlog::info("Global setup started");
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [t%t] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    signal(SIGSEGV, signal_handler);
    signal(SIGABRT, signal_handler);
    return 0;
}
