#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include "layer.hpp"

/******************************************************
 * Constructors
 *****************************************************/

Layer::Layer(int input_size, int output_size, std::string activation_function_name): 
    input_size_(input_size), 
    output_size_(output_size),
    weights_(output_size, input_size), 
    biases_(output_size, 1),
    input_(input_size, 1),
    output_(output_size, 1) {
    
    weights_.randomize();
    biases_.randomize();

    if (compare_ignore_case(activation_function_name, "sigmoid")) {
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
    else {
        throw std::invalid_argument("Invalid activation function name");
    }
}

Layer::Layer(int input_size, int output_size): 
    Layer(input_size, output_size, "sigmoid") {}

/******************************************************
 * Getters
 *****************************************************/

int Layer::get_input_size() const {
    return input_size_;
}

int Layer::get_output_size() const {
    return output_size_;
}

/******************************************************
 * Setters
 *****************************************************/

void Layer::set_activation_function(std::function<Matrix(const Matrix&)> function) {
    activation_function_ = function;
}

void Layer::set_activation_derivative(std::function<Matrix(const Matrix&)> derivative) {
    activation_derivative_ = derivative;
}

/******************************************************
 * Helper functions
 *****************************************************/

bool Layer::compare_ignore_case(const std::string& s1, const std::string& s2) {
    std::string s1_lower = s1;
    std::transform(s1_lower.begin(), s1_lower.end(), s1_lower.begin(), [](unsigned char c) { return std::tolower(c); });
    std::string s2_lower = s2;
    std::transform(s2_lower.begin(), s2_lower.end(), s2_lower.begin(), [](unsigned char c) { return std::tolower(c); });

    return s1_lower == s2_lower;
}
