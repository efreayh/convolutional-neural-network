#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include "activation_function.hpp"

class DenseLayer : public Layer {
public:

    /* Constructors */
    DenseLayer(const int input_size, const int output_size, const std::string& activation_function_name, const double learning_rate);

    /* Layer functionality */
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& output) override;
    
private:
    ActivationFunction function_;
    double learning_rate_;
};

#endif
