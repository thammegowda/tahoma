
#include <tahoma.h>
#include <tahoma/model.h>
#include <tahoma/serialize.h>

namespace tahoma::model {

    auto IModel::get_state() -> Pack {
        Pack state;
        for (auto& pair: named_parameters()) {
            state[pair.key()] = pair.value();
        }
        return state;
    }

    auto IModel::set_state(Pack& state) -> Pack {
        spdlog::info("Restoring model state");
        std::set<std::string> matched_keys;
        std::set<std::string> ignored_keys;
        std::set<std::string> missing_keys;
        for (auto& pair: named_parameters()) {
            auto name = pair.key();
            auto param = pair.value();
            if (state.contains(name)) {
                auto data = std::any_cast<torch::Tensor>(state[name]);
                spdlog::debug("Restoring: {} [{}] ", name, fmt::join(param.sizes().vec(), " x "));
                try{
                    // torch doesnt allow in-place operation on requires_grad=false tensor
                    auto is_grad_required = param.requires_grad();
                    param.set_requires_grad(false);
                    param.copy_(data);
                    param.set_requires_grad(is_grad_required);
                    matched_keys.insert(name);
                } catch (const std::exception& e) {
                    spdlog::error("Failed to restore {}; expected: {} got {}\n{}", name,
                        fmt::join(param.sizes().vec(), " x "),
                        fmt::join(data.sizes().vec(), " x "), e.what());
                    throw e;
                }
            } else {
                spdlog::warn("MISSING: {} [{}] is not found in the checkpoint state", name, fmt::join(param.sizes().vec(), " x "));
                missing_keys.insert(name);
            }
        }
        for (auto& pair: state) {
            if (!matched_keys.contains(pair.first) && !serialize::is_text_key(pair.first)) {
                auto data = std::any_cast<torch::Tensor>(pair.second);
                spdlog::warn("IGNORED: {} [{}] is found but not required", pair.first, fmt::join(data.sizes().vec(), " x "));
                ignored_keys.insert(pair.first);
            }
        }
        return Pack({
            {"matched_keys", matched_keys},
            {"ignored_keys", ignored_keys},
            {"missing_keys", missing_keys}
        });
    }
}