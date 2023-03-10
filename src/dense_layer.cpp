#include "dense_layer.hpp"

/******************************************************
 * Constructors
 *****************************************************/

DenseLayer::DenseLayer(const int input_size, const int output_size, const std::string& activation_function_name, const double learning_rate):
    Layer(input_size, output_size),
    function_(activation_function_name),
    learning_rate_(learning_rate) {}

/******************************************************
 * Layer functionality
 *****************************************************/

Matrix DenseLayer::forward(const Matrix& input) {
    input_ = input;
    z_ = input * weights_ + biases_;
    return function_.apply_function(z_);
}

Matrix DenseLayer::backward(const Matrix& output) {
    Matrix delta = output.element_wise_multiply(function_.apply_derivative(z_));
    Matrix weights_gradient = input_.transpose() * delta;
    Matrix biases_gradient = delta;
    Matrix input_gradient = delta * weights_.transpose();

    weights_ -= weights_gradient.scalar_multiply(learning_rate_);
    biases_ -= biases_gradient.scalar_multiply(learning_rate_);

    return input_gradient;
}
