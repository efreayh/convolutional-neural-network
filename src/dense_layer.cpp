#include "dense_layer.hpp"

/******************************************************
 * Constructors
 *****************************************************/

DenseLayer::DenseLayer(const int input_size, const int output_size, const std::string& activation_function_name, const double learning_rate):
    input_size_(input_size), 
    output_size_(output_size),
    weights_(1, input_size, output_size), 
    biases_(1, 1, output_size),
    input_(1, 1, input_size),
    z_(1, 1, output_size),
    function_(activation_function_name),
    learning_rate_(learning_rate) {
    
    weights_.randomize();
}

/******************************************************
 * Layer functionality
 *****************************************************/

Tensor DenseLayer::forward(const Tensor& input) {
    input_ = input;
    z_ = input * weights_ + biases_;
    return function_.apply_function(z_);
}

Tensor DenseLayer::backward(const Tensor& output) {
    Tensor delta = output.element_wise_multiply(function_.apply_derivative(z_));
    Tensor weights_gradient = input_.transpose() * delta;
    Tensor biases_gradient = delta;
    Tensor input_gradient = delta * weights_.transpose();

    weights_ -= weights_gradient.scalar_multiply(learning_rate_);
    biases_ -= biases_gradient.scalar_multiply(learning_rate_);

    return input_gradient;
}
