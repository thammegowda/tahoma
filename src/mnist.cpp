#include <iostream>
#include <string>

#include <torch/torch.h>
#include <argparse.hpp>


// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register  Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 256));
    fc2 = register_module("fc2", torch::nn::Linear(256, 64));
    fc3 = register_module("fc3", torch::nn::Linear(64, 10));
    //dummy one added but not used
    fc4 = register_module("fc4", torch::nn::ModuleList(
      torch::nn::Linear(64, 10),
      torch::nn::Linear(64, 10))
    );
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = x.reshape({ x.size(0), 784 });

    x = fc1->forward(x);
    x = torch::relu(x);
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());

    x = fc2->forward(x);
    x = torch::relu(x);

    x = fc3->forward(x);
    x = torch::log_softmax(x, /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
  torch::nn::ModuleList fc4{ nullptr };
};


//TODO: get args from command line

int main(int argc, char* argv[]) {


  argparse::ArgumentParser parser("mnist");
  parser.add_argument("data_path")
    .help("path to the MNIST data directory. Run data/get-mnist.sh to download data");

  try {
    parser.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << parser;
    exit(0);
  }

  auto data_path = parser.get<std::string>("data_path");
  //auto data_path = "../data";
  std::cout << "data_path: " << data_path << std::endl;

  // Create a new Net.
  auto net = std::make_shared<Net>();
  auto device = torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  net->to(device);

  // Create a multi-threaded data loader for the MNIST dataset.
  auto data_loader = torch::data::make_data_loader(
    torch::data::datasets::MNIST(data_path).map(
      torch::data::transforms::Stack<>()),
    /*batch_size=*/64);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

  size_t batch_index = 0;
  size_t n_epochs = 10;
  for (size_t epoch = 1; epoch <= n_epochs; ++epoch) {
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *data_loader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      torch::Tensor prediction = net->forward(batch.data.to(device));
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, batch.target.to(device));
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      if (++batch_index % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
          << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        torch::save(net, "net.pt");
      }
    }
  }
  std::cout << "Total Batches:" << batch_index;
}
