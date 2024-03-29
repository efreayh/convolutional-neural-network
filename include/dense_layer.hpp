#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "tensor.hpp"
#include "layer.hpp"

class DenseLayer : public Layer {
public:

    /* Constructors */
    DenseLayer(const int input_size, const int output_size, const double learning_rate);

    /* Layer functionality */
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output) override;
    
private:
    int input_size_;
    int output_size_;
    Tensor weights_;
    Tensor biases_;
    Tensor input_;
    double learning_rate_;
};

#endif
