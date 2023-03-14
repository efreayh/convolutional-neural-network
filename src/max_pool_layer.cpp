#include "max_pool_layer.hpp"
#include "tensor.hpp"

/******************************************************
 * Constructors
 *****************************************************/

MaxPoolLayer::MaxPoolLayer(const int window_size, const int stride):
    window_size_(window_size),
    stride_(stride) {}

/******************************************************
 * Layer functionality
 *****************************************************/

Tensor MaxPoolLayer::forward(const Tensor& input) {
    input_ = input;
    return input.max_pool_forward(window_size_, stride_);
}

Tensor MaxPoolLayer::backward(const Tensor& output) {
    return input_.max_pool_backward(output, window_size_, stride_);
}