#include <string>
#include <cctype>
#include <algorithm>
#include <stdexcept>
#include "utility.hpp"
#include "tensor.hpp"

bool utility::compare_ignore_case(std::string s1, std::string s2) {
    std::transform(s1.begin(), s1.end(), s1.begin(), [](unsigned char c) { return std::tolower(c); });
    std::transform(s2.begin(), s2.end(), s2.begin(), [](unsigned char c) { return std::tolower(c); });

    return s1 == s2;
}

int utility::argmax(const Tensor& input) {
    if (input.get_depth() != 1 || input.get_num_rows() != 1) {
        throw std::invalid_argument("Argmax: invalid input");
    }
    
    int max_index = 0;
    double max_value = input(0)(0,0);

    for (int i = 0; i < input.get_num_columns(); ++i) {
        if (input(0)(0,i) > max_value) {
            max_index = i;
            max_value = input(0)(0,i);
        }
    }

    return max_index;
}

int utility::convolve_result_dim(const int dim, const int filter_dim, const int stride, const std::string& padding_type) {
    if (stride > filter_dim) {
        throw std::invalid_argument("Convolve_result_dim: stride must be less than or equal to filter dimension");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Convolve_result_dim: stride must be greater than 0");
    }
    if (filter_dim > dim) {
        throw std::invalid_argument("Convolve_result_dim: filter dimension must be less or equal to matrix dimension");
    }


    if (utility::compare_ignore_case(padding_type, "full") &&
        filter_dim >= 2 &&
        (dim + filter_dim - 2) % stride == 0) {
        
        return (dim + filter_dim - 2) / stride + 1;
    }
    else if (utility::compare_ignore_case(padding_type, "same")) {
        return dim;
    }
    else if (utility::compare_ignore_case(padding_type, "valid") &&
             (dim - filter_dim) % stride == 0 &&
             (dim != filter_dim || stride >= dim)) {
        
        return (dim - filter_dim) / stride + 1;
    }
    else {
        throw std::invalid_argument("Convolve_result_dim: Failed to calculate convolve result dimension");
    }
}

int utility::max_pool_result_dim(const int dim, const int window_size, const int stride) {
    if (stride > window_size) {
        throw std::invalid_argument("Max_pool_result_dim: stride must be less than or equal to window_size");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Max_pool_result_dim: stride must be greater than 0");
    }
    if (window_size > dim) {
        throw std::invalid_argument("Max_pool_result_dim: window_size must be less or equal to matrix dimensions");
    }

    return ((dim - window_size) + stride - 1) / stride + 1;
}