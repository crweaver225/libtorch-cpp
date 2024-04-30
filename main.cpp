#include <iostream>
#include <torch/torch.h>

struct Dense_Net : torch::nn::Module {
  
    Dense_Net() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 350));
        fc2 = register_module("fc2", torch::nn::Linear(350, 75));
        fc3 = register_module("fc3", torch::nn::Linear(75, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::dropout(x, /*p=*/0.25, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::dropout(x, /*p=*/0.25, /*train=*/is_training());
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

struct Conv_Net : torch::nn::Module {
    public:
        Conv_Net()
            : conv1(torch::nn::Conv2dOptions(1, 16, 3)), // input channels, output channels, kernel size=3
            conv2(torch::nn::Conv2dOptions(16, 32, 3)),
            fc1(800, 128), // Adjusted to 500 based on the calculation
            fc2(128, 10)
        {
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(torch::max_pool2d(conv1->forward(x), 2)); // pool kernel size=2
            x = torch::relu(torch::max_pool2d(conv2->forward(x), 2)); // pool kernel size=2
            x = x.view({-1, 800}); // Flatten to 500 features for fc1
            x = torch::relu(fc1->forward(x));
            x = fc2->forward(x);
            return torch::log_softmax(x, 1);
        }

    private:
        torch::nn::Conv2d conv1, conv2;
        torch::nn::Linear fc1, fc2;
};

int main() {

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout<<"CUDA is available\n";
        device = torch::Device(torch::kCUDA);
    }

    auto net = std::make_shared<Conv_Net>();
    net->to(device);

    auto dataset = torch::data::datasets::MNIST("../mnist_data")
                    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                    .map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(std::move(dataset),  
                        torch::data::DataLoaderOptions().batch_size(64));

    // Set up an optimizer.
    torch::optim::Adam optimizer(net->parameters(), 0.01);

    // Determine how many epochs you want to train for and run them in a loop
    for (size_t epoch = 0; epoch != 5; ++epoch) {

        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto& batch : *data_loader) {

            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = net->forward(data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::nll_loss(prediction, targets);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();

            // Output the loss every 100 batches.
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                    << " | Loss: " << loss.item<float>() << std::endl;
            }
        }
    }


    // Create the test dataset
    auto test_dataset = torch::data::datasets::MNIST("../mnist_data", torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))  // Normalize data
        .map(torch::data::transforms::Stack<>());  // Stack data into a single tensor

    // Create a data loader for the test dataset
    auto test_loader = torch::data::make_data_loader(
        std::move(test_dataset),
        torch::data::DataLoaderOptions());

    // Switch model to evaluation mode

    torch::load(net, "net.pt");

    net->eval();

    int correct = 0;  // Count of correct predictions
    int total = 0;    // Total number of samples processed

    // Iterate over the test dataset
    for (auto& batch : *test_loader) {

        auto data = batch.data.to(device);   // Features (input images)
        auto targets = batch.target.squeeze().to(device); // Targets (true labels)

        // Forward pass to get the output from the model
        auto output = net->forward(data);

        // Get the predictions by finding the index of the max log-probability
        auto pred = output.argmax(1);

        // Compare predictions with true labels
        correct += pred.eq(targets).sum().item<int64_t>();
        total += data.size(0);  // Increment total by the batch size
    }

    // Calculate accuracy
    double accuracy = static_cast<double>(correct) / total;
    std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl;

    if (accuracy * 100 > 95.0) {
        std::cout<<"saving..\n";
        torch::save(net, "net.pt");
    }

    return 0;
}