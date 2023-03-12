#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <string>
#include "tensor.hpp"
#include "layer.hpp"
#include "activation_function.hpp"

class DenseLayer : public Layer {
public:

    /* Constructors */
    DenseLayer(const int input_size, const int output_size, const std::string& activation_function_name, const double learning_rate);

    /* Layer functionality */
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output) override;
    
private:
    int input_size_;
    int output_size_;
    Tensor weights_;
    Tensor biases_;
    Tensor input_;
    Tensor z_;
    ActivationFunction function_;
    double learning_rate_;
};

#endif
