#pragma once
#include <tahoma.h>
#include <torch/torch.h>


/*
Pytorch uses its own serialization format based on ZipFile format.
I could not load a model serialized from pytorch into libtorch's torch::load
I suspect the pytorch relies on some python specific pickling in addition to the zip format.

So, after wasting a lot of time, I decided to use the numpy format to save the model.
There is a script in scripts/ dir that converts pytorch model to numpy format.
*/

namespace tahoma::serialize {
    const vector<string> TEXT_EXTS = {".txt", ".yml", ".yaml", ".json", ".md"};

    auto is_text_key(const string& key) -> bool;

    auto load_npz(std::string path, torch::Device device=torch::kCPU) -> Pack;
    /**
     * @brief Load a model from a npz file
     * 
     * @param path: path to the npz file
     * @return Pack i.e. map of string to torch::Tensor
     */


    auto load_npz(std::string path, std::string key, torch::Device device=torch::kCPU) -> std::any;
    /**
     * @brief Load a single key from a npz file
     * 
     * @param path: path to the npz file
     * @param key: key to load
     * @return T
     */
    

    auto get_npz_keys(std::string path) ->  std::vector<std::string>;
    /**
     * @brief Get all keys in a npz file
     * 
     * @param path: path to the npz file
     * @return std::vector<std::string>
     */

    auto store_npz(std::string path, Pack pack) -> void;
    /**
     * @brief Save a model to a npz file
     * 
     * @param path
     * @param pack
     */
}