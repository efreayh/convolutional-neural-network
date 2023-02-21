#include "dense_layer.hpp"

/******************************************************
 * Constructors
 *****************************************************/

DenseLayer::DenseLayer(int input_size, int output_size, std::function<double(double)> function, std::function<double(double)> derivative): 
    Layer(input_size, output_size, function, derivative) {}

DenseLayer::DenseLayer(int input_size, int output_size): 
    Layer(input_size, output_size) {}

/******************************************************
 * Layer functionality
 *****************************************************/

Matrix DenseLayer::forward(const Matrix& input) {
    input_ = input;
    output_ = weights_ * input_ + biases_;
    return output_.apply_function(activation_function_);
}

Matrix DenseLayer::backward(const Matrix& output_gradient) {
    double learning_rate = 0.001;
    Matrix activation_gradient = output_.apply_function(activation_derivative_);
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
