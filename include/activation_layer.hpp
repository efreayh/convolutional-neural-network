#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <string>
#include "tensor.hpp"
#include "layer.hpp"

class ActivationLayer : public Layer {
public:

    /* Constructors */
    ActivationLayer(const std::string& activation_function_name);

    /* Layer functionality */
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output) override;
    
private:

    Tensor input_;
    std::string activation_function_name_;

    /* Activation functions */
    Tensor sigmoid(const Tensor& in) const;
    Tensor sigmoid_derivative(const Tensor& in) const;
    Tensor relu(const Tensor& in) const;
    Tensor relu_derivative(const Tensor& in) const;
    Tensor softmax(const Tensor& in) const;
    Tensor softmax_derivative(const Tensor& in) const;
};

#endif