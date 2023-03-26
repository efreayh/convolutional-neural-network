#include <string>
#include <stdexcept>
#include <cmath>
#include "dense_layer.hpp"
#include "tensor.hpp"

/******************************************************
 * Constructors
 *****************************************************/

DenseLayer::DenseLayer(const int input_size, const int output_size, const double learning_rate):
    input_size_(input_size), 
    output_size_(output_size),
    weights_(1, input_size, output_size), 
    biases_(1, 1, output_size),
    input_(1, 1, input_size),
    learning_rate_(learning_rate) {
    
    weights_.randomize(0, sqrt(1.0 / input_size));
}

/******************************************************
 * Layer functionality
 *****************************************************/

Tensor DenseLayer::forward(const Tensor& input) {
    if (input.get_depth() != 1) {
        throw std::invalid_argument("DenseLayer forward: tensor must have depth 1");
    }

    input_ = input;
    return input * weights_ + biases_;
}

Tensor DenseLayer::backward(const Tensor& output) {
    if (output.get_depth() != 1) {
        throw std::invalid_argument("DenseLayer forward: tensor must have depth 1");
    }

    Tensor weights_gradient = input_.transpose() * output;
    Tensor biases_gradient = output;
    Tensor input_gradient = output * weights_.transpose();

    weights_ -= weights_gradient.scalar_multiply(learning_rate_);
    biases_ -= biases_gradient.scalar_multiply(learning_rate_);

    return input_gradient;
}
