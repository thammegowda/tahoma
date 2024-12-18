
#pragma once
#include <tahoma.h>
#include <sentencepiece_processor.h>
#include <tahoma/utils.h>

using namespace tahoma;

namespace tahoma::data {

    auto read_lines(const std::string& path) -> Generator<std::string>;
    auto read_lines(const std::vector<std::string>& data_paths) -> Generator<std::vector<std::string>>;

    template <typename T>
    auto vector_to_generator(const std::vector<T>& vec) -> Generator<T> {
        for (const auto& item : vec) {
            co_yield item;
        }
    }

    using RawExample = std::vector<std::string>;
    using IdRawExample = std::pair<size_t, RawExample>;
    struct Example {
        size_t id;
        vector<str> fields;
        vector2d<i32> field_ids;

        ~Example() = default;
        Example(size_t id, vector<str> fields, vector2d<i32> field_ids)
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
            id(other.id), fields(std::move(other.fields)), field_ids(std::move(other.field_ids)) {}

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

    struct DataLoader {
        std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> vocabs;
        config::Config config;

        DataLoader(config::Config config, std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> vocabs):
            config(config), vocabs(vocabs) {}

        DataLoader(config::Config config):
            DataLoader(config, utils::load_vocabs(config["schema"]["vocabs"].as<vector<string>>())) {}

        ~DataLoader() = default;

        auto output_vocab() -> std::shared_ptr<sentencepiece::SentencePieceProcessor>;
        auto make_example(size_t id, RawExample fields, vector<i32> eos_ids, std::vector<size_t> max_lengths, bool max_length_crop=true) -> data::Example;
        auto read_examples(Generator<IdRawExample> rows, std::vector<size_t> max_lengths, bool max_length_crop=true) -> Generator<data::Example>;
        auto read_examples(std::vector<std::string> data_paths, std::vector<size_t> max_lengths, bool max_length_crop=true, size_t num_threads=1) -> Generator<data::Example>;

        //template <typename T>
        auto buffered_shuffle(Generator<Example>& examples, size_t buffer_size) -> Generator<Example>;
        auto make_batches(Generator<Example>& examples, size_t batch_size, bool contiguous = false) -> Generator<Batch>;
        auto get_train_data(size_t n_data_threads=1) -> Generator<Batch>;
        auto get_validation_data() -> Generator<Batch>;
        auto get_samples(std::vector<std::string> data_paths, size_t num_samples) -> Batch;
        auto get_data_sync(std::string dataset_name, std::string fallback_name="trainer") -> Generator<Batch>;
        auto get_data_async(std::string dataset_name, size_t num_threads) -> Generator<Batch>;
    };


    template <typename T>
    auto sample_n_items(const std::vector<T>& buffer, size_t n) -> std::vector<T>;

    auto tensor_shape(Tensor tensor) -> std::string;

    template <typename T>
    auto sample_n_items_stream(Generator<T>& stream, size_t n) -> Generator<T> {
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