
#include <queue>
#include <fstream>
#include <mutex>
#include <thread>
#include <random>
#include <condition_variable>


#include <torch/torch.h>
#include <sentencepiece_processor.h>
#include <tahoma.h>
#include <tahoma/data.h>

#include "./common/queue.cpp"

namespace sp = sentencepiece;
namespace optim = torch::optim;
using namespace tahoma;

namespace tahoma::data {


    // operator<<
    std::ostream& operator<<(std::ostream& os, const Example& ex) {
        os << "Example(" << ex.id << "; fields(" <<
            ex.fields.size() << "): " << ex.fields
            << "; ids(" << ex.field_ids.size() << "):"
            << ex.field_ids.size();
        return os;
    }

    // struct Batch {

    Batch::Batch(std::vector<Example> examples, bool contiguous)
        : examples(examples) {
        if (contiguous) {
            this->contiguous();
        }
    }

    /**
     * Convert a buffer of Examples to a Batch
     * @param examples: a buffer of Examples
     * @return a Batch of tensors
    */
    auto Batch::to_tensors(vector<Example> examples) -> vector<torch::Tensor> {
        /**
         * Convert a buffer of Examples to a Batch
        */
        int32_t batch_size = examples.size();
        assert(batch_size > 0);
        int32_t num_fields = examples[0].field_ids.size();

        vector<int32_t> max_lens(num_fields, 0);
        for (const auto& ex : examples) {
            for (size_t i = 0; i < num_fields; ++i) {
                max_lens[i] = std::max(max_lens[i], (int32_t)ex.field_ids[i].size());
            }
        }

        std::vector<torch::Tensor> fields(num_fields);
        int64_t pad_id = 0;  // pad token id
        for (size_t i = 0; i < num_fields; ++i) {
            fields[i] = torch::full({ batch_size, max_lens[i] }, pad_id, torch::kLong);
            for (int32_t j = 0; j < examples.size(); ++j) {
                fields[i].index_put_({ j, Slice(0, examples[j].field_ids[i].size()) },
                    torch::tensor(examples[j].field_ids[i], torch::kLong));
            }
        }
        return fields;
    }

    auto Batch::contiguous() -> Batch& {
        /**
         * Convert vector of Examples to contiguous tensors with padding
         * The reason we do this on-demand: since batches are queued, we want to avoid filling up memory with padded tensors.
        */
        if (fields.size() == 0) {
            fields = to_tensors(examples);
        } // else already contiguous
        return *this;
    }

    auto Batch::to(torch::Device& device) -> Batch& {
        for (auto idx = 0; idx < fields.size(); ++idx) {
            fields[idx] = fields[idx].to(device);
        }
        return *this;
    }

    auto Batch::size() -> int32_t {
        return examples.size();
    }
    // }; // end of Batch

    //struct DataLoader {  // ideally, DataGenerator but it could be confusing as synthetic data generator

    auto DataLoader::output_vocab() -> std::shared_ptr<sp::SentencePieceProcessor> {
        if (vocabs.empty()) {
            throw std::runtime_error("Vocabs vector is empty");
        }
        return vocabs.back();
    }


    auto DataLoader::make_example(size_t id, std::vector<std::string> fields,
        vector<i32> eos_ids, std::vector<size_t> max_lengths, bool max_length_crop) -> Example {
        i32 num_fields = fields.size();
        auto field_ids = vector2d<int32_t>(num_fields);
        for (size_t i = 0; i < num_fields; ++i) {
            auto ids = vocabs[i]->EncodeAsIds(fields[i]);
            if (eos_ids.size() > i and eos_ids[i] >= 0) {
                ids.push_back(eos_ids[i]);  // append token id
            }
            if (max_length_crop && ids.size() > max_lengths[i]) {
                ids = vector<int32_t>(ids.begin(), ids.begin() + max_lengths[i]);
            }
            field_ids[i] = ids;
        }
        return Example(id, fields, field_ids);
    }

    auto DataLoader::read_examples(Generator<IdRawExample> rows,
        std::vector<size_t> max_lengths, bool max_length_crop, bool add_eos) -> Generator<Example> {
        i32 num_fields = -1;
        vector<i32> eos_ids = {};
        if (add_eos) {
            // iterate through vocabs and get eos_id for each vocab
            for (auto vocab : vocabs) {
                eos_ids.push_back(vocab->eos_id());
            }
        }
        for (auto [row_id, fields] : rows) {
            if (num_fields < 0) {
                num_fields = fields.size(); // initialize on first run
                if (num_fields == 0 || num_fields > vocabs.size()) {
                    throw std::runtime_error(fmt::format("Number of fields must be > 0 and should not exceed the number of vocabs. num_fields: {}, vocabs: {}", num_fields, vocabs.size()));
                }
                if (max_length_crop) {
                    spdlog::debug("Length cropping is enabled. max_length: {}; num_fields: {}", fmt::join(max_lengths, ", "), num_fields);
                    if (max_lengths.size() != num_fields) {
                        throw std::runtime_error(fmt::format("max_length must be of the same size as data_paths. \
                                max_lengths: {}, num_fields: {}", max_lengths.size(), num_fields));
                    }
                }
            }
            if (fields.size() != num_fields) {
                spdlog::warn("All data files must have the same number of fields. \
                        Record number {} has {} fields, expected {}", row_id, fields.size(), num_fields);
                continue;
            }
            auto example = make_example(row_id, fields, eos_ids, max_lengths, max_length_crop);
            if (example.field_ids.empty()) {
                spdlog::warn("Skipping empty record {}", row_id);
                continue;
            }
            co_yield example;
        }
    }

    auto DataLoader::read_examples(std::vector<str> data_paths,
        std::vector<size_t> max_lengths, bool max_length_crop, size_t num_threads, bool add_eos)
        -> Generator<Example> {

        std::vector<i32> eos_ids = {};
        if (add_eos) {
            // iterate through vocabs and get eos_id for each vocab
            for (auto vocab : vocabs) {
                eos_ids.push_back(vocab->eos_id());
            }
        }
        i32 num_fields = -1;
        size_t rec_num = 0;
        auto rows = read_lines(data_paths);
        for (auto fields : rows) {
            if (rec_num == 0) {
                num_fields = fields.size(); // initialize on first run
                if (max_length_crop) {
                    if (max_lengths.size() != num_fields) {
                        throw std::runtime_error("max_length must be of the same size as data_paths");
                    }
                    spdlog::debug("Length cropping is enabled. max_length: {}", fmt::join(max_lengths, ", "));
                }
            }
            rec_num++;
            if (fields.size() != num_fields) {
                spdlog::warn("All data files must have the same number of fields. \
                        Record number {} has {} fields, expected {}", rec_num, fields.size(), num_fields);
                continue;
            }
            auto example = make_example(rec_num, fields, eos_ids, max_lengths, max_length_crop);
            bool skip = fields.empty() || example.field_ids.empty();
            if (skip) {
                spdlog::warn("Skipping empty record {}", rec_num);
                continue;
            }
            co_yield example;
        }
    }

    /**
     * Shuffle examples in a buffer and yield them
     * @param items: a generator of examples
     * @param buffer_size: the size of the buffer
     * @return a generator of shuffled examples
    */
    auto DataLoader::buffered_shuffle(Generator<Example> examples, size_t buffer_size) -> Generator<Example> {
        std::vector<Example> buffer;
        std::vector<size_t> indices(buffer_size);
        std::iota(indices.begin(), indices.end(), 0);
        auto rng = std::default_random_engine{};

        for (auto ex : examples) {
            buffer.push_back(ex);
            if (buffer.size() >= buffer_size) {
                std::shuffle(indices.begin(), indices.end(), rng);
                for (auto idx : indices) {
                    co_yield buffer[idx];
                }
                buffer.clear();
            }
        }
        // the last buffer maybe not full to buffer_size
        if (!buffer.empty()) {
            std::shuffle(buffer.begin(), buffer.end(), rng);
            for (auto ex : buffer) {
                co_yield ex;
            }
        }
    }

    auto DataLoader::sort_examples(Generator<Example> examples, size_t buffer_size,
        size_t sort_field, bool reverse) -> Generator<Example> {
        std::vector<Example> buffer;
        std::vector<size_t> keys(buffer_size);   // sort keys
        std::vector<size_t> indices(buffer_size);
        size_t idx = -1;
        if (buffer_size < 1) {
            throw std::runtime_error("buffer_size must be >= 1, given: " + buffer_size);
        }

        auto compare = [&](size_t i, size_t j) {
            return reverse ? keys[i] > keys[j] : keys[i] < keys[j];
            };

        for (auto ex : examples) {
            idx++;
            buffer.push_back(ex);
            indices[idx] = idx;
            keys[idx] = ex.field_ids[sort_field].size();
            if (buffer.size() == buffer_size) {
                std::sort(indices.begin(), indices.end(), compare);
                for (auto i : indices) {
                    co_yield buffer[i];
                }
                buffer.clear();
                idx = -1;
            }
        }
        // the last buffer maybe not full to buffer_size
        if (!buffer.empty()) {
            std::sort(indices.begin(), indices.begin() + idx, compare);
            for (size_t i = 0; i < idx; ++i) {
                co_yield buffer[indices[i]];
            }
        }
    }

    /**
     * Make batches from examples
     * @param examples: a generator of examples
     * @param batch_size: the size of the batch
     * @param contiguous: whether to make the batch contiguous
     * @return a generator of batches
     * @note: the last batch may be smaller than batch_size
    */
    auto DataLoader::make_batches(Generator<Example> examples, size_t batch_size,
        bool contiguous) -> Generator<Batch> {
        // TODO: buffer and batch equal length examples to reduce padding
        if (batch_size == 0) {
            throw std::runtime_error("batch_size must be > 0");
        }
        auto buffer = std::vector<Example>();
        for (auto ex : examples) {
            buffer.push_back(ex);
            if (buffer.size() >= batch_size) {
                co_yield Batch(buffer, contiguous);
                buffer.clear();
            }
        }
        if (!buffer.empty()) {
            co_yield Batch(buffer, contiguous);
        }
    }

    auto DataLoader::get_train_data(size_t n_data_threads) -> Generator<Batch> {
        if (n_data_threads >= 1) { // asynchronous, on a separate thread
            spdlog::debug("Using async loader with {} data threads", n_data_threads);
            return get_data_async("trainer", n_data_threads);
        } else if (n_data_threads == 0) { // synchronous, on the same thread
            spdlog::debug("Data loading on the main thread");
            return get_data_sync("trainer");
        } else {
            throw std::runtime_error("n_data_threads must be >= 0; given " + n_data_threads);
        }
    }

    auto DataLoader::get_validation_data() -> Generator<Batch> {
        return get_data_sync("validator", "trainer");
    }

    auto DataLoader::get_samples(std::vector<std::string> data_paths, size_t num_samples) -> Batch {
        assert(num_samples > 0);
        vector<IdRawExample> buffer;
        size_t rec_num = 1;
        for (auto line : read_lines(data_paths)) {
            buffer.push_back({ rec_num++, line });
        }
        auto samples = sample_n_items<IdRawExample>(buffer, num_samples);
        auto vector_to_generator = [&samples]() -> Generator<IdRawExample> {
            for (auto sample : samples) {
                co_yield sample;
            }
            };
        auto samples_gen = vector_to_generator();
        auto examples = read_examples(std::move(samples_gen), {}, false);
        auto batches = make_batches(std::move(examples), num_samples);
        for (auto batch : batches) {
            return batch; // first batch
        }
        throw std::runtime_error("No samples found");
    }

    auto DataLoader::get_data_sync(std::string dataset_name, std::string fallback_name) -> Generator<Batch> {
        // TODO remove this once async is stable and bug free
        auto data_paths = config[dataset_name]["data"].as<std::vector<std::string>>();
        // try to locate batch_size in the dataset_name, else fallback to trainer
        auto mini_batch = config[dataset_name]["mini_batch"].as<int>(config[fallback_name]["maxi_batch"].as<int>());
        auto maxi_batch = config[dataset_name]["maxi_batch"].as<int>(config[fallback_name]["mini_batch"].as<int>());
        auto max_length_crop = config[dataset_name]["max_length_crop"].as<bool>(config[fallback_name]["max_length_crop"].as<bool>(true));
        auto max_length = config[dataset_name]["max_length"].as<vector<size_t>>(config[fallback_name]["max_length"].as<vector<size_t>>());

        spdlog::info("Loading data from {}", fmt::join(data_paths, ", "));
        spdlog::info("mini_batch: {}, maxi_batch: {}", mini_batch, maxi_batch);
        spdlog::info("max_length_crop: {}, max_length: {}", max_length_crop, fmt::join(max_length, ", "));

        auto examples = read_examples(data_paths, max_length, max_length_crop);
        examples = buffered_shuffle(std::move(examples), mini_batch * maxi_batch);
        auto batches = make_batches(std::move(examples), mini_batch);
        // CAUTION: direct return of batches generator lead to segfault during coroutine resume
        // we need to co_yield within this function to avoid destruction of coroutines stack 
        // TODO: find a way to forward/relay the generator without the for each co_yield 
        for (auto batch : batches) {
            co_yield batch;
        }
    }


    auto DataLoader::get_data_async(std::string dataset_name, size_t num_threads) -> Generator<Batch> {

        auto loader = MultiThreadedLoader(*this, config[dataset_name], {});
        loader.start(num_threads);

        for (auto batch : loader.generator()) {
            co_yield batch;
        }
        spdlog::info("All workers done");
    }

    auto read_lines_buffered(const std::vector<std::string>& data_paths,
        size_t buffer_size) -> Generator<vector<IdRawExample>> {
        auto rows = read_lines(data_paths);  // generator<vector<string>>
        vector<IdRawExample> buffer;
        size_t rec_num = 0;
        for (auto& line : rows) {
            rec_num++;
            if (line.empty()) {
                spdlog::warn("Empty row found. Skipping");
                continue;
            }
            buffer.push_back({ rec_num, line });
            if (buffer.size() >= buffer_size) {
                co_yield buffer;
                buffer.clear();
            }
        }
        if (!buffer.empty()) {
            co_yield buffer;
        }
        spdlog::info("read_lines_buffered done; reached the end of files");
    };

    MultiThreadedLoader::~MultiThreadedLoader() {
        spdlog::info("Stopping MultiThreadedLoader and {} threads", all_threads.size());
        stop();
        spdlog::info("MultiThreadedLoader stopped");
    }

    void MultiThreadedLoader::start(size_t num_threads) {
        if (all_threads.size() > 0) {
            throw std::runtime_error("Threads already started");
        }
        spdlog::info("Starting data loader with {} threads; \n\tdata_paths: {},\n\tmaxi_batch: {},\n\tmini_batch: {}\n\tmax_length_crop: {}\n\tmax_lengths: {}",
            num_threads, fmt::join(data_paths, ", "), maxi_batch, mini_batch, max_length_crop, fmt::join(max_lengths, ", "));  
        //=================================================//
        auto maxi_batch_task = [&](const std::stop_token& stoken){
            spdlog::info("Reader thread started");
            size_t count = 0;
            for (auto maxi_batch : read_lines_buffered(data_paths, maxi_buffer_size)) {
                if (!input_mappers.empty()) {
                    // map input fields using input_mappers
                    for (auto& [id, ex] : maxi_batch) {
                        for (size_t i = 0; i < std::min(ex.size(), input_mappers.size()); ++i) {
                            ex[i] = input_mappers[i]->map(ex[i]);
                        }
                    }
                }
                //spdlog::debug("Maxi batch {} size: {}. Going to wait for a lock", count, maxi_batch.size());
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&]{ return stoken.stop_requested() || maxi_batch_queue.size() < max_queue_size; });
                    if (stoken.stop_requested()) { break; }
                    //spdlog::debug("Maxi batch thread pushing  {}", count);
                    maxi_batch_queue.push(maxi_batch);
                    count++;
                }
                cv.notify_one();
            }
            {
                std::unique_lock<std::mutex> lock(mutex);
                status.reader_done = true;
            }
            cv.notify_all();
            spdlog::info("Maxi batching thread done");
        };

        //=================================================//
        auto mini_batch_task = [&](const std::stop_token& stoken){
            std::unique_lock<std::mutex> lock(mutex);
            const auto worker_id = ++status.n_workers_started;
            lock.unlock();
            spdlog::info("Worker {} started", worker_id);
            
            size_t count = 0;
            while (true) {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stoken.stop_requested() || !maxi_batch_queue.empty() || status.reader_done; });
                if (stoken.stop_requested() || status.reader_done && maxi_batch_queue.empty()) {
                    break;
                }
                auto maxi_batch = maxi_batch_queue.front();
                maxi_batch_queue.pop();
                lock.unlock();
                cv.notify_all();

                spdlog::debug("worker {}: maxi_batch_count: {} maxi_buffer_size: {}", worker_id, count, maxi_batch.size());
                auto maxi_gen = vector_to_generator<IdRawExample>(std::move(maxi_batch));
                auto examples = loader.read_examples(std::move(maxi_gen), max_lengths, max_length_crop, add_eos);
                if (sort_by == "random") {
                    examples = loader.buffered_shuffle(std::move(examples), maxi_buffer_size);
                } else if (sort_by == "length") {
                    const size_t sort_field = 0; // sort by the first field
                    examples = loader.sort_examples(std::move(examples), maxi_buffer_size, sort_field, /*reverse*/true);
                }

                auto batches = loader.make_batches(std::move(examples), mini_batch);
                for (auto& batch : batches) {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&] { return stoken.stop_requested() || mini_batch_queue.size() < max_queue_size; });
                    if (stoken.stop_requested()) {
                        break;
                    }
                    mini_batch_queue.push(std::move(batch));
                    lock.unlock();
                    cv.notify_all();
                }
            }
            {
                std::unique_lock<std::mutex> lock(mutex);
                status.n_workers_done++;
                spdlog::info("Worker {} done; total workers completed {}", worker_id, status.n_workers_done);
                cv.notify_all();
            }
        };

        auto stoken = stop_source.get_token();
        all_threads.push_back(std::jthread(maxi_batch_task, stoken));
        for (size_t i = 0; i < num_threads; i++) {
            all_threads.push_back(std::jthread(mini_batch_task, stoken));
        }
    }

    void MultiThreadedLoader::stop() {
        // request all threads to stop
        spdlog::info("Requesting all threads to stop");
        for (auto& t : all_threads) {
            t.request_stop();
            // BUG: maxi_batch_task's stoken.stop_requested() sometimes dont get triggered
            // so we are using the below additional stop request which seems to work
            stop_source.request_stop();
        }
        size_t i = 0;
        for (auto& t : all_threads) {
            i++;
            if (t.joinable()) {
                spdlog::debug("Joining worker {} thread {}", i, t.get_id());
                t.join();
            }
        }
    }

    auto MultiThreadedLoader::generator() -> Generator<Batch> {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] {return !mini_batch_queue.empty() || status.is_done(); });
            if (status.is_done() && mini_batch_queue.empty()) {
                break;
            }
            auto batch = mini_batch_queue.front();
            mini_batch_queue.pop();
            lock.unlock();
            cv.notify_one();
            //batch.contiguous();
            co_yield std::move(batch);
        }
    }


    auto read_lines(std::string path) -> Generator<std::string> {
        /**
         * Read lines from a file and yield them one by one.
        */
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            co_yield line;
        }
        file.close();
    }

    auto read_lines(const std::vector<str>& data_paths) -> Generator<RawExample> {
        if (data_paths.empty()) {
            throw std::runtime_error("No data files specified");
        }
        spdlog::debug("Loading data from {}", fmt::join(data_paths, ", "));
        const i32 num_fields = data_paths.size();
        // check that atmost a single "-" is in data_paths
        if (std::count(data_paths.begin(), data_paths.end(), "-") > 1) {
            throw std::runtime_error("At most one '-' (i.e. STDIN) is allowed in data_paths");
        }
        std::vector<std::ifstream> files(num_fields);
        for (size_t i = 0; i < num_fields; ++i) {
            string data_path = data_paths[i];
            if (data_path == "-") {
#ifdef _WIN32
                // https://stackoverflow.com/a/4477381/1506477
                data_path = "CON";
#else
                data_path = "/dev/stdin";
#endif
            }
            files[i].open(data_path);
            if (!files[i]) {
                throw std::runtime_error("Failed to open file " + data_paths[i]);
            }
        }

        bool has_data = true;
        i64 rec_num = 0;
        do {
            auto fields = std::vector<std::string>(num_fields);
            rec_num++;
            for (size_t i = 0; i < num_fields; i++) {
                if (!std::getline(files[i], fields[i])) {
                    has_data = false;
                    break;
                }
                if (fields[i].empty()) {
                    throw std::runtime_error("Empty line in file: " + data_paths[i] + " at line: " + std::to_string(rec_num));
                }
            }

            if (has_data && !fields.empty()) {
                co_yield fields;
            }
        } while (has_data);

        // wish there was a finally{} block to guarantee file closure :(
        for (auto& file : files) {
            file.close();
        }
        spdlog::debug("Reached the end of data files");
    }


    template <typename T>
    auto sample_n_items(const std::vector<T>& buffer, size_t n) -> std::vector<T> {
        std::vector<T> samples = buffer; // copy the original vector
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(samples.begin(), samples.end(), g);
        // If n is greater than the size of the vector, return the whole vector
        if (n > samples.size()) {
            return samples;
        }
        samples.resize(n); // resize the vector to contain only the first n elements
        return samples;
    }

    auto tensor_shape(Tensor tensor) -> std::string {
        /**
         * Return the shape of a tensor as a string.
        */
        std::string shape = "";
        for (auto size : tensor.sizes()) {
            if (shape != "") {
                shape += ", ";
            }
            shape += std::to_string(size);
        }
        return "[" + shape + "]";

    }
}  // namespace data
