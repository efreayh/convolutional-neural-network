#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

class Layer {
public:

    /* Constructors */
    Layer(const int input_size, const int output_size);

    /* Getters */
    int get_input_size() const;
    int get_output_size() const;

    /* Layer functionality */
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output) = 0;

protected:
    int input_size_;
    int output_size_;
    Tensor weights_;
    Tensor biases_;
    Tensor input_;
    Tensor z_;
};

#endif
