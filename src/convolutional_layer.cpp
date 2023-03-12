#include <string>
#include "convolutional_layer.hpp"
#include "tensor.hpp"
#include "activation_function.hpp"
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
    input_depth_(input_depth),
    input_rows_(input_rows),
    input_columns_(input_columns),
    filter_rows_(filter_rows),
    filter_columns_(filter_columns),
    stride_(1),
    filters_(output_depth, Tensor(input_depth, filter_rows, filter_columns)),
    biases_(output_depth,
            utility::convolve_result_dim(input_rows, filter_rows, 1, "valid"),
            utility::convolve_result_dim(input_columns, filter_columns, 1, "valid")),
    learning_rate_(learning_rate) {
    
    for (int i = 0; i < output_depth; ++i) {
        filters_[i].randomize();
    }
} 

/******************************************************
 * Layer functionality
 *****************************************************/

Tensor ConvolutionalLayer::forward(const Tensor& input) {
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
