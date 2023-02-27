#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include "activation_function.hpp"

class DenseLayer : public Layer {
public:

    /* Constructors */
    DenseLayer(int input_size, int output_size, std::string activation_function_name, double learning_rate);

    /* Layer functionality */
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& output) override;
    
private:
    ActivationFunction function_;
    double learning_rate_;
};

#endif
