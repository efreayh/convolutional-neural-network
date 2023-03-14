#ifndef MAX_POOL_LAYER_HPP
#define MAX_POOL_LAYER_HPP

#include "tensor.hpp"
#include "layer.hpp"

class MaxPoolLayer : public Layer {
public:

    /* Constructors */
    MaxPoolLayer(const int window_size, const int stride);

    /* Layer functionality */
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output) override;
    
private:
    int window_size_;
    int stride_;
    Tensor input_;
};

#endif