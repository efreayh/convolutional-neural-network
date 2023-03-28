#include <iostream>
#include <chrono>
#include <memory>
#include "utility.hpp"
#include "tensor.hpp"
#include "dense_layer.hpp"
#include "convolutional_layer.hpp"
#include "activation_layer.hpp"
#include "max_pool_layer.hpp"
#include "flatten_layer.hpp"
#include "mnist_data_set.hpp"
#include "neural_network.hpp"

int main() {

    double learning_rate = 0.1;
    int epochs = 10;

    std::cout << "Loading data set..." << std::endl ;

    MNISTDataSet dataset("data/mnist.csv");
    
    std::unique_ptr<Layer> layer0 = std::make_unique<ConvolutionalLayer>(16, 1, 28, 28, 3, 3, learning_rate);
    std::unique_ptr<Layer> layer1 = std::make_unique<ActivationLayer>("relu");
    std::unique_ptr<Layer> layer2 = std::make_unique<MaxPoolLayer>(2, 2);
    std::unique_ptr<Layer> layer3 = std::make_unique<ConvolutionalLayer>(16 * 2, 16, 13, 13, 3, 3, learning_rate);
    std::unique_ptr<Layer> layer4 = std::make_unique<ActivationLayer>("relu");
    std::unique_ptr<Layer> layer5 = std::make_unique<MaxPoolLayer>(2, 2);
    std::unique_ptr<Layer> layer6 = std::make_unique<FlattenLayer>(16 * 2, 6, 6);
    std::unique_ptr<Layer> layer7 = std::make_unique<DenseLayer>(16 * 2 * 6 * 6, 100, learning_rate);
    std::unique_ptr<Layer> layer8 = std::make_unique<ActivationLayer>("sigmoid");
    std::unique_ptr<Layer> layer9 = std::make_unique<DenseLayer>(100, 10, learning_rate);
    std::unique_ptr<Layer> layer10 = std::make_unique<ActivationLayer>("sigmoid");

    NeuralNetwork network;
    network.add_layer(std::move(layer0));
    network.add_layer(std::move(layer1));
    network.add_layer(std::move(layer2));
    network.add_layer(std::move(layer3));
    network.add_layer(std::move(layer4));
    network.add_layer(std::move(layer5));
    network.add_layer(std::move(layer6));
    network.add_layer(std::move(layer7));
    network.add_layer(std::move(layer8));
    network.add_layer(std::move(layer9));
    network.add_layer(std::move(layer10));

    std::cout << "Starting training..." << std::endl;
 
    for (int epoch = 0; epoch < epochs; ++epoch) {

        std::cout << "************ Epoch " << (epoch + 1) << "/" << epochs << " ************" << std::endl;

        auto beg = std::chrono::high_resolution_clock::now();

        // Train
        for (int i = 0; i < dataset.get_train_size(); ++i) {

            std::cout << "Training iteration: " << (i + 1) << "/" << dataset.get_train_size() << std::flush;

            Tensor tensor_in = dataset.get_train_data(i);
            Tensor expected_out = dataset.get_train_label(i);

            network.train(tensor_in, expected_out);

            std::cout << "\r";
        }

        std::cout << std::endl;

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Predicting..." << std::endl;

        int num_correct = 0;

        // Test
        for (int i = 0; i < dataset.get_test_size(); ++i) {

            Tensor tensor_in = dataset.get_test_data(i);
            Tensor expected_out = dataset.get_test_label(i);

            Tensor result = network.predict(tensor_in);

            int predicted = utility::argmax(result);
            int expected = utility::argmax(expected_out);
            
            if (predicted == expected) {
                ++num_correct;
            }
        }

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);

        double accuracy = (double) num_correct / dataset.get_test_size();

        std::cout << "Accuracy: " << (accuracy * 100) << "% Time: " << duration.count() << "ms" << std::endl;
    }

    return 0;
}
