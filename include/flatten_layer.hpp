#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "tensor.hpp"
#include "layer.hpp"

class FlattenLayer : public Layer {
public:

    /* Constructors */
    FlattenLayer(const int input_depth, const int input_rows, const int input_columns);

    /* Layer functionality */
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output) override;

private:
    int input_depth_;
    int input_rows_;
    int input_columns_;
};

#endif