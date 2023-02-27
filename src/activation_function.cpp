#include <cctype>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "activation_function.hpp"

/******************************************************
 * Constructors
 *****************************************************/

ActivationFunction::ActivationFunction(const std::string& function_name) {
    if (compare_ignore_case(function_name, "sigmoid")) {
        set_sigmoid();
    }
    else {
        throw std::invalid_argument("ActivationFunction: invalid function_name");
    }
}

/******************************************************
 * Core functionality
 *****************************************************/

Matrix ActivationFunction::apply_function(const Matrix& input_matrix) const {
    return activation_function_(input_matrix);
}

Matrix ActivationFunction::apply_derivative(const Matrix& input_matrix) const {
    return activation_derivative_(input_matrix);
}

/******************************************************
 * Activation function type setters
 *****************************************************/

void ActivationFunction::set_sigmoid() {
    activation_function_ = [](const Matrix& input_matrix) -> Matrix {
        Matrix result(input_matrix.get_num_rows(), 1);

        for (int i = 0; i < input_matrix.get_num_rows(); ++i) {
            double sigmoid = 1 / (1 + exp(-input_matrix(i, 0)));
            result(i, 0) = sigmoid;
        }

        return result;
    };

    activation_derivative_ = [](const Matrix& input_matrix) -> Matrix {
        Matrix result(input_matrix.get_num_rows(), 1);

        for (int i = 0; i < input_matrix.get_num_rows(); ++i) {
            double sigmoid = 1 / (1 + exp(-input_matrix(i, 0)));
            double sigmoid_derivative = sigmoid * (1 - sigmoid);
            result(i, 0) = sigmoid_derivative;
        }

        return result;
    };
}

/******************************************************
 * Helper functions
 *****************************************************/

bool ActivationFunction::compare_ignore_case(const std::string& s1, const std::string& s2) {
    std::string s1_lower = s1;
    std::transform(s1_lower.begin(), s1_lower.end(), s1_lower.begin(), [](unsigned char c) { return std::tolower(c); });
    std::string s2_lower = s2;
    std::transform(s2_lower.begin(), s2_lower.end(), s2_lower.begin(), [](unsigned char c) { return std::tolower(c); });

    return s1_lower == s2_lower;
}