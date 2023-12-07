#pragma once
#include <iostream>
#include <memory>
#include <chrono>

#include "../common/config.hpp"

namespace nn = torch::nn;
namespace optim = torch::optim;

using namespace std;
using namespace chrono;

namespace rtg::train {

    struct StatsCounter {
        int64_t step_num = 0;
        int64_t tot_sents = 0;
        int64_t tot_tokens = 0;
        double tot_loss = 0.0;

        string log_frequency = "";
        int64_t log_frequency_step = -1;
        int64_t log_frequency_tokens = -1;
        int64_t log_frequency_time_sec = -1;

        int64_t last_log_step = 0;
        int64_t last_log_tokens = 0;
        chrono::time_point<chrono::high_resolution_clock> last_log_time = chrono::high_resolution_clock::now();
        chrono::time_point<chrono::high_resolution_clock> start_time = chrono::high_resolution_clock::now();


        auto set_log_frequency(string arg) {  // log as in logging (not logarithm)
            /*
            This is inspired by Marian NMT.
            arg should be an integer with a suffix
            suffix u =>  number of steps i.e updates
            suffix t => number of tokens
            the second last character is k,M,B (optional), then it is interpreted as kilo, million, billion
            suffix s,m,h =>  number of seconds, minutes, hours
            */
            log_frequency = arg;
            assert(arg.size() > 0);
            auto suffix = arg.back();
            auto num = arg.substr(0, arg.size() - 1);
            int64_t scaler = 1;
            // check if the second last character is k,M,B
            if (suffix == 'u' || suffix == 't') {
                switch (num.size() > 0 ? num[num.size() - 1] : '?') {
                case 'k':  // proper
                case 'K': // typo
                    scaler = 1'000;
                    break;
                case 'M': // proper
                    scaler = 1'000'000;
                    break;
                case 'B': // proper
                case 'G': // okay
                    scaler = 1'000'000'000;
                    break;
                }
                if (scaler > 1) {
                    num = num.substr(0, num.size() - 1);
                }
            }

            size_t num_val = std::stoi(num);
            switch (suffix) {
            case 'u': // updates
                log_frequency_step = scaler * num_val;
                break;
            case 't': // tokens
                log_frequency_tokens = scaler * num_val;
                break;
            case 's': // seconds
                log_frequency_time_sec = num_val;
                break;
            case 'm': // minutes
                log_frequency_time_sec = num_val * 60;
                break;
            case 'h': // hours
                log_frequency_time_sec = num_val * 60 * 60;
                break;
            default:
                throw runtime_error(fmt::format("Invalid log frequency argument {}", arg));
            }
        }

        // add constructor
        StatsCounter() {}
        StatsCounter(string log_frequency) {
            set_log_frequency(log_frequency);
        }

        // copy and move
        StatsCounter(const StatsCounter& other) = default;
        StatsCounter(StatsCounter&& other) = default;
        StatsCounter& operator=(const StatsCounter& other) = default;
        StatsCounter& operator=(StatsCounter&& other) = default;

        double avg_loss() {
            return step_num > 0 ? tot_loss / step_num : 0.0;
        }

        auto update(double loss, size_t num_sents, size_t num_tokens, size_t num_steps = 1) -> StatsCounter& {
            tot_sents += num_sents;
            tot_tokens += num_tokens;
            step_num += num_steps;
            tot_loss += loss;
            bool log_now = false;
            if (log_frequency_step > 0 && step_num - last_log_step >= log_frequency_step) {
                log_now = true;
                last_log_step = step_num;
            }
            else if (log_frequency_tokens > 0 && num_tokens - last_log_tokens >= log_frequency_tokens) {
                log_now = true;
                last_log_tokens = num_tokens;
            }
            else if (log_frequency_time_sec > 0 &&
                chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - last_log_time).count() >= log_frequency_time_sec) {
                log_now = true;
                last_log_time = chrono::high_resolution_clock::now();
            }
            if (log_now) {
                auto duration_ms = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time);
                auto toks_rate = 1000.0f * tot_tokens / duration_ms.count();
                auto sents_rate = 1000.0f * tot_sents / duration_ms.count();
                spdlog::info("Step: {}; Loss: {:.5f}; AvgLoss: {:.5f}; sents: {}; toks: {}, speed: {:.1f} tok/s {:.1f} sent/s",
                    step_num, loss, avg_loss(), tot_sents, tot_tokens, toks_rate, sents_rate);
            }
            return *this;
        }
    };
}