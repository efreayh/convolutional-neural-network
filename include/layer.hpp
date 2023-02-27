#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"

class Layer {
public:

    /* Constructors */
    Layer(int input_size, int output_size);

    /* Getters */
    int get_input_size() const;
    int get_output_size() const;

    /* Layer functionality */
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& output) = 0;

protected:
    int input_size_;
    int output_size_;
    Matrix weights_;
    Matrix biases_;
    Matrix input_;
    Matrix z_;
};

#endif
