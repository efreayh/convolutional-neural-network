#ifndef LAYER_HPP
#define LAYER_HPP

#include <functional>
#include "matrix.hpp"

class Layer {
public:

    /* Constructors */
    Layer(int input_size, int output_size, std::function<double(double)> function, std::function<double(double)> derivative);
    Layer(int input_size, int output_size);

    /* Getters */
    int get_input_size() const;
    int get_output_size() const;

    /* Setters */
    void set_activation_function(std::function<double(double)> function);
    void set_activation_derivative(std::function<double(double)> derivative);

    /* Layer functionality */
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& output_gradient) = 0;

protected:
    int input_size_;
    int output_size_;
    Matrix weights_;
    Matrix biases_;
    Matrix input_;
    Matrix output_;
    std::function<double(double)> activation_function_;
    std::function<double(double)> activation_derivative_;
};

#endif
