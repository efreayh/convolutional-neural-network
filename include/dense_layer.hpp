#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"

class DenseLayer : public Layer {
public:

    /* Constructors */
    DenseLayer(int input_size, int output_size, std::string activation_function_name);
    DenseLayer(int input_size, int output_size);

    /* Layer functionality */
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& output_gradient) override;
    
    Matrix get_weights() const;
};

#endif
