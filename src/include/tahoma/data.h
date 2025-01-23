
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
        vector<std::string> fields;
        vector2d<i32> field_ids;

        ~Example() = default;
        Example(size_t id, vector<str> fields, vector2d<i32> field_ids)
            : id(id), fields(fields), field_ids(field_ids) {        
}

        //copy ctr
        Example(const Example& other)
            : id(other.id), fields(other.fields), field_ids(other.field_ids) {        
}

        //copy assignment
        Example& operator=(const Example& other) {
            if (this != &other) {
                id = other.id; fields = other.fields; field_ids = other.field_ids;
            }
            return *this;
        }

        // move ctr
        Example(Example&& other) noexcept :
            id(other.id), fields(std::move(other.fields)), field_ids(std::move(other.field_ids)) {
        }

        //std::ostream& operator<<(std::ostream& os, const Example& example);

        friend std::ostream& operator<<(std::ostream& os, const Example& example);
    };
    std::ostream& operator<<(std::ostream& os, const Example& ex);

    struct Batch {
        std::vector<Example> examples;
        std::vector<Tensor> fields;

        Batch(std::vector<Example> examples, bool contiguous = false);
        
        ~Batch()=default;
        static auto to_tensors(std::vector<Example> examples) -> std::vector<torch::Tensor>;
        auto contiguous() -> Batch&;
        auto to(torch::Device& device) -> Batch&;
        auto size() -> i32;
        void clear(){
            examples.clear();
            fields.clear();
        }
    };

    struct DataLoader {
        config::Config config;
        std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> vocabs;

        DataLoader(config::Config config, std::vector<std::shared_ptr<sentencepiece::SentencePieceProcessor>> vocabs) :
            config(config), vocabs(vocabs) {
        }

        DataLoader(config::Config config) :
            DataLoader(config, utils::load_vocabs(config["schema"]["vocabs"].as<vector<string>>())) {
        }

        ~DataLoader() = default;

        auto output_vocab() -> std::shared_ptr<sentencepiece::SentencePieceProcessor>;
        auto make_example(size_t id, RawExample fields, vector<i32> eos_ids, std::vector<size_t> max_lengths, bool max_length_crop = true) -> data::Example;
        auto read_examples(Generator<IdRawExample> rows, std::vector<size_t> max_lengths, bool max_length_crop = true, bool add_eos = true) -> Generator<data::Example>;
        auto read_examples(std::vector<std::string> data_paths, std::vector<size_t> max_lengths, bool max_length_crop = true, size_t num_threads = 1, bool add_eos = true) -> Generator<data::Example>;

        //template <typename T>
        auto buffered_shuffle(Generator<Example> examples, size_t buffer_size) -> Generator<Example>;
        auto sort_examples(Generator<Example> examples,  size_t buffer_size, std::function<int64_t(const Example&)> key_func, bool reverse=false) -> Generator<Example>;
        auto make_batches(Generator<Example> examples, size_t batch_size, bool contiguous = false) -> Generator<Batch>;
        auto get_train_data(size_t n_data_threads = 1) -> Generator<Batch>;
        auto get_validation_data() -> Generator<Batch>;
        auto get_samples(std::vector<std::string> data_paths, size_t num_samples) -> Batch;
        auto get_data_sync(std::string dataset_name, std::string fallback_name = "trainer") -> Generator<Batch>;
        auto get_data_async(std::string dataset_name, size_t num_threads) -> Generator<Batch>;
    };


    struct LineMapper {

        virtual auto map(const std::string& line) -> std::string = 0;
        
        auto operator()(const std::string& line) -> std::string {
            return map(line);
        }
    };


    struct MultiThreadedLoader {
        /*
        File reading is done on a single thread
        the rest of work is done on multiple worker threads
        */

        struct StatusContainer {
            bool reader_done = false;
            size_t n_workers_started = 0;
            size_t n_workers_done = 0;

            StatusContainer() = default;
            StatusContainer(StatusContainer&&) = delete;
            StatusContainer(const StatusContainer&) = delete;
            StatusContainer& operator=(const StatusContainer&) = delete;

            bool is_done() {
                return reader_done && n_workers_started > 0 && n_workers_done == n_workers_started;
            }
        };

        DataLoader& loader;
        std::vector<std::string> data_paths;
        size_t mini_batch;
        size_t maxi_batch;
        size_t maxi_buffer_size;
        bool max_length_crop;
        std::vector<size_t> max_lengths;
        bool add_eos = true;
        StatusContainer status;

        string sort_by = ""; // random, length
        std::queue<vector<IdRawExample>> maxi_batch_queue;
        std::queue<Batch> mini_batch_queue;
        size_t max_queue_size = 32;
        std::mutex mutex;
        std::condition_variable cv;

        std::vector<std::jthread> all_threads;
        std::vector<Ptr<LineMapper>> input_mappers;
        std::stop_source stop_source;

        MultiThreadedLoader(DataLoader& loader, std::vector<std::string> data_paths, size_t mini_batch, size_t maxi_batch,
            bool max_length_crop, std::vector<size_t> max_lengths, std::vector<Ptr<LineMapper>> input_mappers)
            :
            loader(loader),
            data_paths(data_paths),
            mini_batch(mini_batch),
            maxi_batch(maxi_batch),
            maxi_buffer_size(maxi_batch* mini_batch),
            max_length_crop(max_length_crop),
            max_lengths(max_lengths),
            input_mappers(input_mappers)
        {}

        MultiThreadedLoader(DataLoader& loader, YAML::Node config, std::vector<Ptr<LineMapper>> input_mappers) :
            MultiThreadedLoader(loader,
                config["data"].as<std::vector<std::string>>(),
                config["mini_batch"].as<size_t>(),
                config["maxi_batch"].as<size_t>(1),
                config["max_length_crop"].as<bool>(false),
                config["max_length"].as<std::vector<size_t>>(),
                input_mappers)
        {}

        MultiThreadedLoader() = delete;
        MultiThreadedLoader(MultiThreadedLoader&&) = delete;
        MultiThreadedLoader(const MultiThreadedLoader&) = delete;
        MultiThreadedLoader& operator=(const MultiThreadedLoader&) = delete;
        ~MultiThreadedLoader();

        void start(size_t num_threads);
        void stop();
        auto generator() -> Generator<Batch>;

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