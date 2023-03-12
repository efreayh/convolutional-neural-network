#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <string>
#include "tensor.hpp"

namespace utility {
    bool compare_ignore_case(std::string s1, std::string s2);
    int argmax(const Tensor& input);
    int convolve_result_dim(const int dim, const int filter_dim, const int stride, const std::string& padding_type);
    int max_pool_result_dim(const int dim, const int window_size, const int stride);
}

#endif