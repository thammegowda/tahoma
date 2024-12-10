
#include <tahoma.h>
#include <tahoma/serialize.h>
#include <cnpy.h>

namespace tahoma::serialize {
    // TODO: support loading a single key from the npz file
    const auto LOOKUP_TORCH_TYPE = std::map<std::pair<char, size_t>, torch::ScalarType>{
            {{'f', 2}, torch::kHalf},
            {{'f', 4}, torch::kFloat},
            {{'f', 8}, torch::kDouble},
            // todo: bf16
            {{'i', 1}, torch::kInt8},
            {{'i', 2}, torch::kInt16},
            {{'i', 4}, torch::kInt32},
            {{'i', 8}, torch::kInt64},

            {{'u', 1}, torch::kUInt8},
            {{'u', 2}, torch::kUInt16},
            {{'u', 4}, torch::kUInt32},
            {{'u', 8}, torch::kUInt64},

            {{'b', 1}, torch::kBool},
            {{'c', 1}, torch::kChar},
        };

    auto is_text_key(const string& key) -> bool {
        return std::any_of(TEXT_EXTS.begin(), TEXT_EXTS.end(),
            [&key](const string& ext) { return key.size() >= ext.size() && key.rfind(ext) == key.size() - ext.size();}
        );
    }

    auto load_npz(std::string path, torch::Device device) -> Pack {
        spdlog::info("Loading model from {}", path);
        auto archive = cnpy::npz_load(path);   // map of string to NpyArray
        Pack pack;
        for (auto& [key, val_ptr] : archive) {
            cnpy::NpyArray val = *val_ptr;
            if (is_text_key(key)) { // convert utf8 bytes to string
                std::string text(val.bytes.begin(), val.bytes.end());
                pack[key] = text;
            } else {
                // vec<size_t> -> vec<int64_t> -> IntArrayRef
                auto shape_vec = std::vector<i64>(val.shape.begin(), val.shape.end());
                if (!LOOKUP_TORCH_TYPE.contains({val.type, val.word_size})) {
                    throw std::runtime_error(fmt::format("Unsupported type: {}{}", val.type, val.word_size));
                }
                auto dtype = LOOKUP_TORCH_TYPE.at({val.type, val.word_size});
                Tensor tensor = torch::zeros(at::IntArrayRef(shape_vec), at::TensorOptions(device).dtype(dtype));
                std::memcpy(tensor.data_ptr(), val.data(), val.size());
                pack[key] = tensor;
            }
        }
        return pack;
    }

    auto list_npz_keys(std::string path) -> std::vector<std::string> {
        //TODO: this is not efficient, we are loading the entire archive
        auto archive = cnpy::npz_load(path);
        std::vector<std::string> keys;
        for (auto& [key, val] : archive) {
            keys.push_back(key);
        }
        return keys;
    }

    void store_npz(std::string path, Pack pack) {
        cnpy::NpyArray arr;
        size_t count = 0;
        spdlog::info("Saving model to {}", path);
        vector<cnpy::NpzItem> items;
        throw std::runtime_error("Not implemented");
        /*
        for (auto& [key, val] : pack) {
            string mode = count++ == 0 ? "w" : "a";
            if (is_text_key(key)) {
                auto text = std::any_cast<std::string>(val);
                const auto utf8_bytes = std::vector<uint8_t>(text.begin(), text.end());
                const auto shape = std::vector<unsigned long>{ utf8_bytes.size() };
                const string key2 = key;
                //cnpy::npz_save(path, key, data, shape, mode);
                auto item = cnpy::NpzItem(key2, utf8_bytes, shape);
                items.push_back(item);
            } else {
                // for now, assuming all vals to be float tensors
                auto tensor = std::any_cast<torch::Tensor>(val);
                auto shape_int = tensor.sizes().vec();
                auto shape = std::vector<size_t>(shape_int.begin(), shape_int.end());
                auto data = tensor.cpu().contiguous().data_ptr<float>();
                //cnpy::npz_save(path, key, data, shape, mode);
                auto item = cnpy::NpzItem(key, std::vector<float>(data, data + tensor.numel()), shape);
                items.push_back(item);
            }
        }
        */
    }
}