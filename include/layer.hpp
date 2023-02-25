#ifndef LAYER_HPP
#define LAYER_HPP

#include <string>
#include <functional>
#include "matrix.hpp"

class Layer {
public:

    /* Constructors */
    Layer(int input_size, int output_size, std::string activation_function_name);
    Layer(int input_size, int output_size);

    /* Getters */
    int get_input_size() const;
    int get_output_size() const;

    /* Setters */
    void set_activation_function(std::function<Matrix(const Matrix&)> function);
    void set_activation_derivative(std::function<Matrix(const Matrix&)> derivative);

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
    std::function<Matrix(const Matrix&)> activation_function_;
    std::function<Matrix(const Matrix&)> activation_derivative_;

    /* Helper functions */
    bool compare_ignore_case(const std::string& s1, const std::string& s2);
};

#endif
