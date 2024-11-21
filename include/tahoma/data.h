
#pragma once
#include <tahoma.h>
#include <sentencepiece_processor.h>

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
        auto make_example(std::vector<std::string> fields, vector<i32> eos_ids, std::vector<size_t> max_lengths, bool max_length_crop=true) -> data::Example;
        auto read_examples(Generator<std::vector<std::string>> rows, std::vector<size_t> max_lengths, bool max_length_crop=true) -> Generator<data::Example>;
        auto read_examples(std::vector<std::string> data_paths, std::vector<size_t> max_lengths, bool max_length_crop=true, i32 num_threads=1) -> Generator<data::Example>;

        //template <typename T>
        auto buffered_shuffle(Generator<data::Example>& examples, size_t buffer_size) -> Generator<data::Example>;
        auto make_batches(Generator<data::Example>& examples, size_t batch_size, bool contiguous = false) -> Generator<data::Batch>;
        auto get_train_data(size_t n_data_threads=1) -> Generator<data::Batch>;
        auto get_validation_data() -> Generator<data::Batch>;
        auto get_samples(std::vector<std::string> data_paths, i32 num_samples) -> data::Batch;
        auto get_data_sync(std::string dataset_name, std::string fallback_name="trainer") -> Generator<data::Batch>;
        auto get_data_async(std::string dataset_name, i32 num_threads) -> Generator<data::Batch>;
        auto get_data_async_new(std::string dataset_name, i32 num_threads) -> Generator<data::Batch>;
    };


    template <typename T>
    auto sample_n_items(const std::vector<T>& buffer, i32 n) -> std::vector<T>;

    auto tensor_shape(Tensor tensor) -> std::string;

    template <typename T>
    auto sample_n_items_stream(Generator<T>& stream, i32 n) -> Generator<T> {
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