#include "dense_layer.hpp"

/******************************************************
 * Constructors
 *****************************************************/

DenseLayer::DenseLayer(int input_size, int output_size, std::string activation_function_name): 
    Layer(input_size, output_size, activation_function_name) {}

DenseLayer::DenseLayer(int input_size, int output_size): 
    Layer(input_size, output_size) {}

/******************************************************
 * Layer functionality
 *****************************************************/

Matrix DenseLayer::forward(const Matrix& input) {
    input_ = input;
    output_ = weights_ * input_ + biases_;
    return activation_function_(output_);
}

Matrix DenseLayer::backward(const Matrix& output_gradient) {
    double learning_rate = 0.001;
    Matrix activation_gradient = activation_derivative_(output_);
    Matrix input_gradient = weights_.transpose() * output_gradient.element_wise_multiply(activation_gradient);
    Matrix weights_gradient = output_gradient.element_wise_multiply(activation_gradient) * input_.transpose();
    Matrix biases_gradient = output_gradient.element_wise_multiply(activation_gradient);
    weights_ -= weights_gradient.scalar_multiply(learning_rate);
    biases_ -= biases_gradient.scalar_multiply(learning_rate);
    return input_gradient;
}

Matrix DenseLayer::get_weights() const {
    return weights_;
}
