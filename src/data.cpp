
#include <queue>
#include <fstream>
#include <mutex>
#include <thread>
#include <random>
#include <condition_variable>

#include <__generator.hpp>  //reference implementation of generator
#include <torch/torch.h>
#include <sentencepiece_processor.h>
#include <tahoma.h>
#include <tahoma/config.h>
#include <tahoma/data.h>
#include <tahoma/utils.h>

namespace nn = torch::nn;
namespace optim = torch::optim;
namespace fs = std::filesystem;
namespace sp = sentencepiece;


using namespace tahoma;

namespace tahoma::data {


    // operator<<
    std::ostream& operator<<(std::ostream& os, const Example& ex) {
        os << "Example(" << ex.id << "; fields(" <<
            ex.fields.size() << "): " << ex.fields
            << "; ids: (" << ex.field_ids.size() << "))";
        return os;
    }

    /*
    std::ostream& Example::operator<<(std::ostream& os, const Example& example) {
        return os << "Example(" << example.id << "; fields(" <<
            example.fields.size() << "): " << example.fields
            << "; ids: (" << example.field_ids.size() << "))";
    }
    */

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
        auto Batch::to_tensors(vector<Example> examples) -> vector<torch::Tensor>{
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

        void Batch::contiguous() {
            /**
             * Convert vector of Examples to contiguous tensors with padding
             * The reason we do this on-demand: since batches are queued, we want to avoid filling up memory with padded tensors.
            */
            if (fields.size() > 0) {
                // already contiguous
                return;
            }
            auto temp = to_tensors(examples);
            fields.insert(fields.end(), temp.begin(), temp.end());
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

    auto load_vocabs(const config::Config& config) -> vector<std::shared_ptr<sp::SentencePieceProcessor>> {
        auto vocab_paths = config["schema"]["vocabs"].as<std::vector<std::string>>();
        assert(!vocab_paths.empty()); // expected atleast one vocabulary
        // SentencePieceProcessor is not copyable and movable, so we use pointers
        vector<std::shared_ptr<sp::SentencePieceProcessor>> spps;
        for (auto vocab_path : vocab_paths) {
            spdlog::debug("loading vocab {}", vocab_path);
            auto spp = std::make_shared<sp::SentencePieceProcessor>();
            if (!fs::exists(vocab_path)) {
                spdlog::error("Vocab file {} not found", vocab_path);
                throw std::runtime_error("Vocab file " + vocab_path + " not found");
            }
            if (!spp->Load(vocab_path).ok()) {
                throw std::runtime_error("Unable to load vocab from " + vocab_path);
            }
            spps.push_back(spp);
        }
        return spps;
    }


    //struct DataLoader {  // ideally, DataGenerator but it could be confusing as synthetic data generator

        inline auto DataLoader::output_vocab() -> std::shared_ptr<sp::SentencePieceProcessor> {
            if (vocabs.empty()) {
                throw std::runtime_error("Vocabs vector is empty");
            }
            return vocabs.back();
        }

        auto DataLoader::read_lines(std::vector<str> data_paths) -> std::generator<std::vector<std::string>> {
             if (data_paths.empty()) {
                throw std::runtime_error("No data files specified");
            }
            spdlog::debug("Loading data from {}", fmt::join(data_paths, ", "));
            const i32 num_fields = data_paths.size();
            std::vector<std::ifstream> files(num_fields);
            for (size_t i = 0; i < num_fields; ++i) {
                files[i].open(data_paths[i]);
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

                if (has_data) {
                    co_yield fields;
                }
            } while (has_data);

            // wish there was a finally{} block to guarantee file closure :(
            for (auto& file : files) {
                file.close();
            }
            spdlog::debug("Reached the end of data files");
        }

        auto DataLoader::read_examples(std::generator<vector<std::string>> rows, vector<size_t> max_length,
                bool max_length_crop) -> std::generator<data::Example> {
            i32 num_fields = -1;
            i64 rec_num = 0;
            vector<i32> eos_ids = {};
            // iterate through vocabs and get eos_id for each vocab
            for (auto vocab : vocabs) {
                eos_ids.push_back(vocab->eos_id());
            }

            for (auto fields: rows) {
                if (num_fields < 0) {
                    num_fields = fields.size(); // initialize on first run
                    if (max_length_crop){
                        if(max_length.size() != num_fields) {
                            throw std::runtime_error("max_length must be of the same size as data_paths");
                        }
                        spdlog::debug("Length cropping is enabled. max_length: {}", fmt::join(max_length, ", "));
                    }
                }
                if (fields.size() != num_fields) {
                    throw std::runtime_error(fmt::format("All data files must have the same number of fields. \
                        Record number {} has {} fields, expected {}", rec_num, fields.size(), num_fields));
                }
                bool skip = false;
                auto field_ids = vector2d<int32_t>(num_fields);
                 for (size_t i = 0; i < num_fields; ++i) {
                    auto ids = vocabs[i]->EncodeAsIds(fields[i]);
                    ids.push_back(eos_ids[i]);  //  append token id
                    if (max_length_crop && ids.size() > max_length[i]) {
                        ids = vector<int32_t>(ids.begin(), ids.begin() + max_length[i]);
                    }
                    skip = skip || ids.empty();
                    field_ids[i] = ids;
                }
                if (skip) {
                    spdlog::warn("Skipping empty record {}", rec_num);
                    continue;
                }
                co_yield data::Example(rec_num, fields, field_ids);
                rec_num++;
            }
        }

        /**
         * Shuffle examples in a buffer and yield them
         * @param examples: a generator of examples
         * @param buffer_size: the size of the buffer
         * @return a generator of shuffled examples
        */
        auto DataLoader::buffered_shuffle(std::generator<data::Example> examples, size_t buffer_size) -> std::generator<data::Example> {
            vector<data::Example> buffer;
            for (auto ex : examples) {
                buffer.push_back(ex);
                if (buffer.size() >= buffer_size) {
                    std::shuffle(buffer.begin(), buffer.end(), std::mt19937(std::random_device()()));
                    for (auto ex : buffer) {
                        co_yield ex;
                    }
                    buffer.clear();
                }
            }
            if (!buffer.empty()) {
                std::shuffle(buffer.begin(), buffer.end(), std::mt19937(std::random_device()()));
                for (auto ex : buffer) {
                    co_yield ex;
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
        auto DataLoader::make_batches(std::generator<data::Example> examples, size_t batch_size,
            bool contiguous) -> std::generator<data::Batch> {
            // TODO: buffer and batch equal length examples to reduce padding
            vector<data::Example> buffer;
            for (auto ex : examples) {
                buffer.push_back(ex);
                if (buffer.size() >= batch_size) {
                    co_yield data::Batch(buffer, contiguous);
                    buffer = vector<data::Example>();
                }
            }
            if (!buffer.empty()) {
                co_yield data::Batch(buffer, contiguous);
            }
        }

        auto DataLoader::get_train_data(size_t n_data_threads) -> std::generator<data::Batch> {
            if (n_data_threads >= 1) { // asynchronous, on a separate thread
                spdlog::debug("Using async loader with {} data threads", n_data_threads);
                // TODO: support multiple threads
                return get_data_async("trainer");
            } else if (n_data_threads == 0) { // synchronous, on the same thread
                spdlog::debug("Data loading on the main thread");
                return get_data_sync("trainer");
            } else {
                throw std::runtime_error("n_data_threads must be >= 1");
            }
        }

        auto DataLoader::get_validation_data() -> std::generator<data::Batch> {
            return get_data_sync("validator", "trainer");
        }

        auto DataLoader::get_samples(std::vector<std::string> data_paths, i32 num_samples) -> data::Batch {
            assert (num_samples > 0);
            auto samples = tahoma::utils::sample_n_items<vector<std::string>>(read_lines(data_paths), num_samples);
            auto examples = read_examples(std::move(samples), {}, false);
            auto batches = make_batches(std::move(examples), num_samples);
            for (auto batch: batches){
                return batch; // first batch
            }
            throw std::runtime_error("No samples found");
        }

        auto DataLoader::get_data_sync(std::string dataset_name, std::string fallback_name) -> std::generator<data::Batch> {
            // TODO remove this once async is stable and bug free
            auto data_paths = config[dataset_name]["data"].as<std::vector<std::string>>();
            // try to locate batch_size in the dataset_name, else fallback to trainer
            auto mini_batch = config[dataset_name]["mini_batch"].as<int>(config[fallback_name]["maxi_batch"].as<int>());
            auto maxi_batch = config[dataset_name]["maxi_batch"].as<int>(config[fallback_name]["mini_batch"].as<int>());
            auto max_length_crop = config[dataset_name]["max_length_crop"].as<bool>(config[fallback_name]["max_length_crop"].as<bool>(true));
            auto max_length = config[dataset_name]["max_length"].as<vector<size_t>>(config[fallback_name]["max_length"].as<vector<size_t>>());
            auto lines = read_lines(data_paths);
            auto examples = read_examples(std::move(lines), max_length, max_length_crop);
            examples = buffered_shuffle(std::move(examples), mini_batch * maxi_batch);
            auto batches = make_batches(std::move(examples), mini_batch);
            return batches;
        }

        auto DataLoader::get_data_async(std::string dataset_name) -> std::generator<data::Batch> {

            auto data_paths = this->config[dataset_name]["data"].as<std::vector<std::string>>();
            auto mini_batch = this->config[dataset_name]["mini_batch"].as<i32>();
            auto maxi_batch = this->config[dataset_name]["maxi_batch"].as<i32>(1);
            auto max_length_crop = this->config[dataset_name]["max_length_crop"].as<bool>(true);
            auto max_length = this->config[dataset_name]["max_length"].as<vector<size_t>>();

            std::mutex mutex;
            std::condition_variable cv;
            bool producer_done = false;
            std::queue<data::Batch> queue;
            size_t max_queue_size = 24;
            size_t thread_sleep_ms = 4;  // in ms

            std::thread producer_thread([&] {
                auto lines = read_lines(data_paths);
                auto examples = read_examples(std::move(lines), max_length, max_length_crop);
                examples = buffered_shuffle(std::move(examples), mini_batch * maxi_batch);
                auto batches = make_batches(std::move(examples), mini_batch);
                for (auto batch : batches) {
                    while (queue.size() >= max_queue_size) {
                        // wait for the queue to drain
                        std::this_thread::sleep_for(std::chrono::milliseconds(thread_sleep_ms));
                    }
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        queue.push(batch);
                    }
                    cv.notify_one();
                }
                // notify consumers that we are done
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    producer_done = true;
                }
            });

            // producer_thread.detach();
            // read from queue and co_yield
            while (true) {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] {return !queue.empty() || producer_done; });
                if (queue.empty() && producer_done) {
                    break;
                }
                auto batch = queue.front();
                queue.pop();
                lock.unlock();
                //batch.contiguous();
                co_yield batch;
            }
            producer_thread.join();
        }
    // }; // end of DataLoader

}