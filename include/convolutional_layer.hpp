#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include <string>
#include <vector>
#include "tensor.hpp"
#include "layer.hpp"

class ConvolutionalLayer : public Layer {
public:

    /* Constructors */
    ConvolutionalLayer(const int output_depth,
                       const int input_depth,
                       const int input_rows,
                       const int input_columns,
                       const int filter_rows,
                       const int filter_columns,
                       const double learning_rate);

    /* Layer functionality */
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output) override;
    
private:
    int output_depth_;
    int input_depth_;
    int input_rows_;
    int input_columns_;
    int filter_rows_;
    int filter_columns_;
    int stride_;
    Tensor input_;
    Tensor output_;
    std::vector<Tensor> filters_;
    Tensor biases_;
    double learning_rate_;
};

#endif