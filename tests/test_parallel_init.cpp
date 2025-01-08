#include <tahoma.h>
#include <tahoma/utils.h>

/**
 * @TG: Initialization of models in parallel threads seems to be slow.
 * I want to initialize large models on 8 GPUs using 8 threads, and it takes 8x the time of initializing on 1 GPU.
 *  There seemed to be thread locking issue. Since pytorch never really needs true multi-threaded parallelism,
 *  they might have done it that way to avoid bugs. So here I experiment and benchmark multi-threaded model initialization.
 */
// Asked this on pytorch forum:  https://discuss.pytorch.org/t/x/215093

namespace tahoma::tests {

    const torch::Device DEVICE = torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU); 

    struct NetImpl : nn::Module {
        nn::Sequential layers;

        NetImpl(std::vector<int64_t> sizes, torch::Device device = DEVICE):
            layers{ register_module("layers", torch::nn::Sequential()) } {
            for (size_t i = 0; i < sizes.size() - 1; i++) {
                layers->push_back(nn::Linear(sizes[i], sizes[i + 1]));
                layers->push_back(nn::Functional(torch::relu));
            }
            this->to(device);
        }

        Tensor forward(Tensor x) {
            x = layers->forward(x);
            return x;
        }
    };
    TORCH_MODULE(Net);


    int test_parallel_init(std::vector<std::string> args){
        // deep network; FFN with a lot of layers to make it deep
        spdlog::info("torch version {}", TORCH_VERSION);
        std::vector<int64_t> dims = { 
            1024, 4096, 8192, 16384, 8192, 4096, 1024, 512, 256, 512,
            1024, 4096, 8192, 16384, 8192, 4096, 1024, 512, 256, 512,
            1024, 4096, 8192, 16384, 8192, 4096, 1024, 512, 256, 512,
            1024, 4096, 8192, 16384, 8192, 4096, 1024, 512, 256, 512,
            1024, 4096, 8192, 16384, 8192, 4096, 1024, 512, 256, 512,
            };

        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available");
        }
        std::vector<torch::Device> devices;
        for (auto i = 0; i < torch::cuda::device_count(); i++) {
            devices.push_back(torch::Device(torch::kCUDA, i));
        }
        double time_1th, time_mth;
        { // scope for timer 
            auto timer1 = utils::Timer("[1-threaded initializer]");
            auto model = Net(dims, devices[0]);
            time_1th = timer1.elapsed();
        }
        size_t n_threads = std::min(devices.size(), 4ul);
        { // scope for timer 
            auto timer1 = utils::Timer(fmt::format("[{}-threaded initializer]", n_threads));
            std::vector<std::jthread> threads;
            for (int i = 0; i < n_threads; i++) {
                auto t = std::jthread([i, &dims, &devices] {
                    auto device = devices[i];
                    auto timer2 = utils::Timer(fmt::format("{}", device.str()));
                    auto model = Net(dims, device);
                });
                threads.push_back(std::move(t));
            }
            for (auto& t : threads) {
                t.join();
            }
            time_mth = timer1.elapsed();
        }
        // multi threaded init should not be more than 20% slower than single threaded
        return time_mth <= 1.2 * time_1th ? 0 : 1; 
    }
}