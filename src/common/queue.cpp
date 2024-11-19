#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>
#include <cassert>

namespace tahoma::common {


    template <typename T>
    class BoundedQueue {
        /**
         * A bounded queue that blocks when full.
         */
    protected:
        std::queue<T> queue;
        std::mutex mutex;
        std::condition_variable cv;
        size_t max_size;
        std::atomic<int> n_producers = 0;
        size_t n_items = 0;

    public:
        BoundedQueue(size_t max_size) : max_size(max_size) {
            assert(max_size > 0);
        }

        void push(T item) {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return queue.size() < max_size; });
            queue.push(item);
            n_items++;
            lock.unlock();
            cv.notify_one();
        }

        std::optional<T> pop() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return !queue.empty() || n_producers == 0; });
            if (queue.empty() && n_producers == 0) {
                return std::nullopt;
            }
            auto item = queue.front();
            queue.pop();
            lock.unlock();
            cv.notify_one();
            return item;
        }

        bool empty() {
            std::unique_lock<std::mutex> lock(mutex);
            return queue.empty();
        }

        size_t size() {
            std::unique_lock<std::mutex> lock(mutex);
            return queue.size();
        }

        void producer_start() {
            std::unique_lock<std::mutex> lock(mutex);
            n_producers++;
        }

        void producer_done() {
            std::unique_lock<std::mutex> lock(mutex);
            n_producers--;
            cv.notify_all();
        }

        void await_producer_start() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return n_producers > 0; });
        }


        bool is_producing() {
            std::unique_lock<std::mutex> lock(mutex);
            return n_producers > 0;
        }

    };
}
