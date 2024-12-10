#include <tahoma.h>
#include <tahoma/serialize.h>
#include <tahoma/utils.h>
#include <yaml-cpp/yaml.h>

using namespace tahoma;

namespace tahoma::tests {

    int test_npz_load(std::vector<std::string> args) {
        auto model_file = args[0];

        auto pack = serialize::load_npz(model_file);
        std::cerr << "Loaded model from " << model_file << "\n n keys:" << pack.size() << std::endl;
        if (pack.empty()) {
            std::cerr << "Error: Failed to load model from " << model_file << std::endl;
            return 1;
        }
        auto meta_text = std::any_cast<string>(pack["meta.yml"]);
        auto meta = YAML::Load(meta_text);
        int num_errs = 0;
        for (auto& [key, val] : pack) {
            if (utils::ends_with(key, { ".yml", ".yaml", ".json" })) {
                auto text = std::any_cast<std::string>(val);
                // no validation for now
            } else {
                float expected = meta[key].as<float>();
                auto tensor = std::any_cast<Tensor>(val);
                float got = tensor.sum().item<float>();
                float eps = 1e-6;
                if (std::abs(expected - got) > eps) {
                    num_errs++;
                    std::cerr << key << " expected: " << expected << ", got: " << got << std::endl;
                    std::cerr << tensor << std::endl;
                }
            }
        }
        return num_errs;
    }

}  // namespace tahoma::tests