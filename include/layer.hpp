#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

class Layer {
public:

    /* Layer functionality */
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output) = 0;

};

#endif
