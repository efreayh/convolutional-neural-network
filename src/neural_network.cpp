#include <vector>
#include <memory>
#include "tensor.hpp"
#include "layer.hpp"
#include "neural_network.hpp"

/******************************************************
 * Constructors
 *****************************************************/

NeuralNetwork::NeuralNetwork(): num_layers_(0) {}

/******************************************************
 * Setters
 *****************************************************/

void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
    ++num_layers_;
}

/******************************************************
 * Operations
 *****************************************************/

void NeuralNetwork::train(const Tensor& input, const Tensor& expected_output) {
    Tensor result = input;

    for (int i = 0; i < num_layers_; ++i) {
        result = layers_[i]->forward(result);
    }

    result = result - expected_output;

    for (int i = num_layers_ - 1; i >= 0; --i) {
        result = layers_[i]->backward(result);
    }
}

Tensor NeuralNetwork::predict(const Tensor& input) {
    Tensor result = input;

    for (int i = 0; i < num_layers_; ++i) {
        result = layers_[i]->forward(result);
    }

    return result;
}
