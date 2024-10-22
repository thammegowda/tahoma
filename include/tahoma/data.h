
#pragma once
#include <tahoma.h>
#include <tahoma/config.h>

using namespace tahoma;

namespace tahoma::data {

    auto read_lines(std::string path) -> std::generator<std::string>;

    struct Example {
        i64 id;
        vector<str> fields;
        vector2d<i32> field_ids;

        ~Example() = default;
        Example(i64 id, vector<str> fields, vector2d<i32> field_ids)
            : id(id), fields(fields), field_ids(field_ids) {}

        //copy ctr
        Example(const Example& other)
            : id(other.id), fields(other.fields), field_ids(other.field_ids) {}

        //copy assignment
        Example& operator=(const Example& other) {
            if (this != &other) {
                id = other.id; fields = other.fields; field_ids = other.field_ids;
            }
            return *this;
        }

        // move ctr
        Example(Example&& other) noexcept :
            id(other.id), fields(other.fields), field_ids(other.field_ids) {}

        //std::ostream& operator<<(std::ostream& os, const Example& example);

        friend std::ostream& operator<<(std::ostream& os, const Example& example);
    };
    std::ostream& operator<<(std::ostream& os, const Example& ex);

    struct Batch {
        std::vector<Example> examples;
        std::vector<Tensor> fields;

        Batch(std::vector<Example> examples, bool contiguous = false);
        static auto to_tensors(std::vector<Example> examples) -> std::vector<torch::Tensor>;
        void contiguous();
        auto to(torch::Device& device) -> Batch&;
        auto size() -> i32;
        ~Batch() = default;
    };

    auto load_vocabs(const config::Config& config) -> std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>>;

    struct DataLoader {
        std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> vocabs;
        config::Config config;

        DataLoader(config::Config config, std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> vocabs):
            config(config), vocabs(vocabs) {}
        DataLoader(config::Config config):
            DataLoader(config, load_vocabs(config)) {}

        ~DataLoader() = default;

        auto output_vocab() -> std::shared_ptr<sentencepiece::SentencePieceProcessor>;
        auto read_lines(std::vector<std::string> data_paths) -> std::generator<std::vector<std::string>>;
        auto read_examples(std::generator<std::vector<std::string>> rows, std::vector<size_t> max_length, bool max_length_crop=true) -> std::generator<data::Example>;
        auto buffered_shuffle(std::generator<data::Example> examples, size_t buffer_size) -> std::generator<data::Example>;
        auto make_batches(std::generator<data::Example> examples, size_t batch_size, bool contiguous = false) -> std::generator<data::Batch>;
        auto get_train_data(size_t n_data_threads=1) -> std::generator<data::Batch>;
        auto get_validation_data() -> std::generator<data::Batch>;
        auto get_samples(std::vector<std::string> data_paths, i32 num_samples) -> data::Batch;
        auto get_data_sync(std::string dataset_name, std::string fallback_name="trainer") -> std::generator<data::Batch>;
        auto get_data_async(std::string dataset_name) -> std::generator<data::Batch>;
    };


    template <typename T>
    auto sample_n_items(const std::vector<T>& buffer, i32 n) -> std::vector<T>;

    auto tensor_shape(Tensor tensor) -> std::string;

    template <typename T>
    auto sample_n_items_stream(std::generator<T>& stream, i32 n) -> std::generator<T> {
        // buffer -> sample -> yield
        std::vector<std::vector<std::string>> buffer;
        for (auto item : stream) {
            buffer.push_back(item);
        }
        auto samples = sample_n_items(buffer, n);
        for (auto sample : samples) {
            co_yield sample;
        }
    }

}  // namespace data