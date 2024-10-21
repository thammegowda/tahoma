#pragma once
#include <iostream>
#include <memory>
#include <chrono>
#include <tahoma.h>

namespace optim = torch::optim;
using namespace tahoma;

namespace tahoma::train {

    struct StatsCounter {
        i64 step_num = 0;
        i64 tot_sents = 0;
        i64 tot_tokens = 0;
        f64 tot_loss = 0.0;

        std::string log_frequency = "";
        i64 log_frequency_step = -1;
        i64 log_frequency_tokens = -1;
        i64 log_frequency_time_sec = -1;
        i64 log_first = 0;

        i64 last_log_step = 0;
        i64 last_log_tokens = 0;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_log_time = std::chrono::high_resolution_clock::now();
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
        std::string name = "";

        StatsCounter() = default;
        StatsCounter(std::string log_frequency, std::string name = "", i64 log_first = 0)
            : name(name), log_first(log_first) {
            set_log_frequency(log_frequency);
        }

        StatsCounter(const StatsCounter& other) = default;
        StatsCounter(StatsCounter&& other) = default;
        StatsCounter& operator=(const StatsCounter& other) = default;
        StatsCounter& operator=(StatsCounter&& other) = default;

        f32 avg_loss();
        auto set_log_frequency(std::string arg) -> void;
        auto update(f32 loss, size_t num_sents, size_t num_tokens, f32 lr, size_t num_steps = 1) -> StatsCounter&;
    };
}
