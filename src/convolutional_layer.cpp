#include <string>
#include <stdexcept>
#include <cmath>
#include "convolutional_layer.hpp"
#include "tensor.hpp"
#include "utility.hpp"

/******************************************************
 * Constructors
 *****************************************************/

ConvolutionalLayer::ConvolutionalLayer(const int output_depth,
                                       const int input_depth,
                                       const int input_rows,
                                       const int input_columns,
                                       const int filter_rows,
                                       const int filter_columns,
                                       const double learning_rate):
    output_depth_(output_depth),
    output_rows_(utility::convolve_result_dim(input_rows, filter_rows, 1, "valid")),
    output_columns_(utility::convolve_result_dim(input_columns, filter_columns, 1, "valid")),
    input_depth_(input_depth),
    input_rows_(input_rows),
    input_columns_(input_columns),
    filter_rows_(filter_rows),
    filter_columns_(filter_columns),
    stride_(1),
    filters_(output_depth, Tensor(input_depth, filter_rows, filter_columns)),
    biases_(output_depth, output_rows_, output_columns_),
    learning_rate_(learning_rate) {
    
    for (int i = 0; i < output_depth; ++i) {
        filters_[i].randomize(0, sqrt(2.0 / (filter_rows * filter_columns * input_depth)));
    }
} 

/******************************************************
 * Layer functionality
 *****************************************************/

Tensor ConvolutionalLayer::forward(const Tensor& input) {
    if (input.get_depth() != input_depth_ || input.get_num_rows() != input_rows_ || input.get_num_columns() != input_columns_) {
        throw std::invalid_argument("ConvolutionalLayer forward: invalid input dimensions");
    }

    input_ = input;
    output_ = biases_;

    for (int i = 0; i < output_depth_; ++i) {
        for (int j = 0; j < input_depth_; ++j) {
            output_(i) += input(j).correlate(filters_[i](j), stride_, "valid");
        }
    }

    return output_;
}

Tensor ConvolutionalLayer::backward(const Tensor& output) {
    if (output.get_depth() != output_depth_ || output.get_num_rows() != output_rows_ || output.get_num_columns() != output_columns_) {
        throw std::invalid_argument("ConvolutionalLayer forward: invalid input dimensions");
    }

    std::vector<Tensor> filters_gradient(output_depth_, Tensor(input_depth_, filter_rows_, filter_columns_));
    Tensor input_gradient(input_depth_, input_rows_, input_columns_);

    for (int i = 0; i < output_depth_; ++i) {
        for (int j = 0; j < input_depth_; ++j) {
            filters_gradient[i](j) = input_(j).correlate(output(i), stride_, "valid");
            input_gradient(j) += output(i).convolve(filters_[i](j), stride_, "full");
        }
    }

    for (int i = 0; i < output_depth_; ++i) {
        filters_[i] -= filters_gradient[i].scalar_multiply(learning_rate_);
    }
    biases_ -= output.scalar_multiply(learning_rate_);
    return input_gradient;
}
