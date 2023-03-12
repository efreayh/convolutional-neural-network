#include <stdexcept>
#include <cmath>
#include "activation_function.hpp"
#include "utility.hpp"

/******************************************************
 * Constructors
 *****************************************************/

ActivationFunction::ActivationFunction(const std::string& function_name) {
    if (utility::compare_ignore_case(function_name, "sigmoid")) {
        set_sigmoid();
    }
    else {
        throw std::invalid_argument("ActivationFunction: invalid function_name");
    }
}

/******************************************************
 * Core functionality
 *****************************************************/

Tensor ActivationFunction::apply_function(const Tensor& input_tensor) const {
    Tensor result;
    for (int i = 0; i < input_tensor.get_depth(); ++i) {
        result.append_matrix(activation_function_(input_tensor(i)));
    }
    return result;
}

Tensor ActivationFunction::apply_derivative(const Tensor& input_tensor) const {
    Tensor result;
    for (int i = 0; i < input_tensor.get_depth(); ++i) {
        result.append_matrix(activation_derivative_(input_tensor(i)));
    }
    return result;
}

/******************************************************
 * Activation function type setters
 *****************************************************/

void ActivationFunction::set_sigmoid() {
    activation_function_ = [](const Matrix& input_matrix) -> Matrix {
        if (input_matrix.get_num_rows() != 1) {
            throw std::invalid_argument("sigmoid: invalid matrix dimensions");
        }

        Matrix result(1, input_matrix.get_num_columns());

        for (int i = 0; i < input_matrix.get_num_columns(); ++i) {
            double sigmoid = 1 / (1 + exp(-input_matrix(0, i)));
            result(0, i) = sigmoid;
        }

        return result;
    };

    activation_derivative_ = [](const Matrix& input_matrix) -> Matrix {
        if (input_matrix.get_num_rows() != 1) {
            throw std::invalid_argument("sigmoid: invalid matrix dimensions");
        }

        Matrix result(1, input_matrix.get_num_columns());

        for (int i = 0; i < input_matrix.get_num_columns(); ++i) {
            double sigmoid = 1 / (1 + exp(-input_matrix(0, i)));
            double sigmoid_derivative = sigmoid * (1 - sigmoid);
            result(0, i) = sigmoid_derivative;
        }

        return result;
    };
}
