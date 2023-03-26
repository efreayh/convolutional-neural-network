#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "activation_layer.hpp"
#include "tensor.hpp"
#include "utility.hpp"

/******************************************************
 * Constructors
 *****************************************************/

ActivationLayer::ActivationLayer(const std::string& activation_function_name) {
    if (!utility::compare_ignore_case(activation_function_name, "sigmoid") &&
        !utility::compare_ignore_case(activation_function_name, "relu") &&
        !utility::compare_ignore_case(activation_function_name, "softmax")) {
        
        throw std::invalid_argument("ActivationLayer constructor: invalid activation_function_name provided");
    }

    activation_function_name_ = activation_function_name;
}

/******************************************************
 * Layer functionality
 *****************************************************/

Tensor ActivationLayer::forward(const Tensor& input) {
    input_ = input;

    if (utility::compare_ignore_case(activation_function_name_, "sigmoid")) {
        return sigmoid(input);
    }
    else if (utility::compare_ignore_case(activation_function_name_, "relu")) {
        return relu(input);
    }
    else {
        return softmax(input);
    }
}

Tensor ActivationLayer::backward(const Tensor& output) {
    if (utility::compare_ignore_case(activation_function_name_, "sigmoid")) {
        return output.element_wise_multiply(sigmoid_derivative(input_));
    }
    else if (utility::compare_ignore_case(activation_function_name_, "relu")) {
        return output.element_wise_multiply(relu_derivative(input_));
    }
    else {
        return output.element_wise_multiply(softmax_derivative(input_));
    }
}

/******************************************************
 * Activation functions
 *****************************************************/

Tensor ActivationLayer::sigmoid(const Tensor& in) const {
    Tensor result (in.get_depth(), in.get_num_rows(), in.get_num_columns());

    for (int i = 0; i < in.get_depth(); ++i) {
        for (int j = 0; j < in.get_num_rows(); ++j) {
            for (int k = 0; k < in.get_num_columns(); ++k) {
                double sigmoid = 1.0 / (1 + std::exp(-in(i)(j, k)));
                result(i)(j, k) = sigmoid;
            }
        }
    }
    
    return result;
}

Tensor ActivationLayer::sigmoid_derivative(const Tensor& in) const {
    Tensor result (in.get_depth(), in.get_num_rows(), in.get_num_columns());

    for (int i = 0; i < in.get_depth(); ++i) {
        for (int j = 0; j < in.get_num_rows(); ++j) {
            for (int k = 0; k < in.get_num_columns(); ++k) {
                double sigmoid = 1.0 / (1 + std::exp(-in(i)(j, k)));
                double sigmoid_derivative = sigmoid * (1 - sigmoid);
                result(i)(j, k) = sigmoid_derivative;
            }
        }
    }
    
    return result;
}

Tensor ActivationLayer::relu(const Tensor& in) const {
    Tensor result (in.get_depth(), in.get_num_rows(), in.get_num_columns());

    for (int i = 0; i < in.get_depth(); ++i) {
        for (int j = 0; j < in.get_num_rows(); ++j) {
            for (int k = 0; k < in.get_num_columns(); ++k) {
                result(i)(j, k) = std::max(0.0, in(i)(j, k));
            }
        }
    }
    
    return result;
}

Tensor ActivationLayer::relu_derivative(const Tensor& in) const {
    Tensor result (in.get_depth(), in.get_num_rows(), in.get_num_columns());

    for (int i = 0; i < in.get_depth(); ++i) {
        for (int j = 0; j < in.get_num_rows(); ++j) {
            for (int k = 0; k < in.get_num_columns(); ++k) {
                if (in(i)(j, k) <= 0.0) {
                    result(i)(j, k) = 0.0;
                }
                else {
                    result(i)(j, k) = 1.0;
                }
            }
        }
    }
    
    return result;
}

Tensor ActivationLayer::softmax(const Tensor& in) const {
    double sum = 0.0;
    Tensor result (in.get_depth(), in.get_num_rows(), in.get_num_columns());

    for (int i = 0; i < in.get_depth(); ++i) {
        for (int j = 0; j < in.get_num_rows(); ++j) {
            for (int k = 0; k < in.get_num_columns(); ++k) {
                double exponent = std::exp(in(i)(j, k));
                result(i)(j, k) = exponent;
                sum += exponent;
            }
        }
    }
    
    return result.scalar_multiply(1.0 / sum);
}

Tensor ActivationLayer::softmax_derivative(const Tensor& in) const {
    throw std::logic_error("Unimplemented");
}
