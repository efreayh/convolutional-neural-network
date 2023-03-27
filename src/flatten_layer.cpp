#include "tensor.hpp"
#include "flatten_layer.hpp"

/******************************************************
 * Constructors
 *****************************************************/

FlattenLayer::FlattenLayer(const int input_depth, const int input_rows, const int input_columns): 
    input_depth_(input_depth),
    input_rows_(input_rows),
    input_columns_(input_columns) {}

/******************************************************
 * Layer functionality
 *****************************************************/

Tensor FlattenLayer::forward(const Tensor& input) {
    return input.flatten();
}

Tensor FlattenLayer::backward(const Tensor& output) {
    Tensor input = output;
    input.reshape(input_depth_, input_rows_, input_columns_);
    return input;
}
