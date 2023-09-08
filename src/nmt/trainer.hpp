#include <iostream>
//#include <concepts>
#include <coroutine>
#include <torch/torch.h>
#include "../common/utils.hpp"

namespace nn = torch::nn;
namespace optim = torch::optim;

namespace rtg::trainer {


    struct TrainerOptions {
        std::vector<std::string> data_paths;
        std::vector<std::string> vocab_paths;
        int64_t epochs;
        int64_t batch_size;
    };

    struct Batch{
        int64_t fields;
        //std::vector<int64_t>& fields;
        //std::vector<std::vector<int64_t>> fields;

        //Batch(std::vector<int64_t>& fields) : fields(fields) {}
        //~Batch() { delete &fields;}
    };

    class Trainer {
    protected:
        nn::AnyModule& model;
        optim::Optimizer& optimizer;
        optim::LRScheduler& scheduler;
        nn::AnyModule& criterion;

    public:
        Trainer(nn::AnyModule model, optim::Optimizer& optimizer, optim::LRScheduler& scheduler, nn::AnyModule criterion) :
            model(model), optimizer(optimizer), scheduler(scheduler), criterion(criterion) {
            std::cout << "Trainer constructor\n";
        }

        ~Trainer() {
            std::cout << "Trainer destructor\n";
        }

        auto get_train_data(TrainerOptions& options) -> rtg::utils::Generator<Batch> {
            std::cout << "Trainer get_train_data\n";
            auto batch_size = options.batch_size;
           for (int i =0 ; i < 10; i++){
                //std::vector<int64_t> fields = {i, i+1, i+2};
                int64_t fields =i; 
                co_yield Batch(fields);
                //co_yield Batch { .fields = i };
                i++;
            }
        }

        void train(TrainerOptions& options) {
            std::cout << "Trainer train\n";
            auto train_data = get_train_data(options);

            int64_t step_num = 0;
            for (int64_t epoch = 0; epoch < options.epochs; epoch++) {
            for (int j = 0; train_data; j++){
                    Batch batch = train_data();
                    std::cout << "epoch: " << epoch << " step: " << step_num << " src: " << batch.fields << "\n";
                    step_num++;
                }
            }
        }
    };

}




