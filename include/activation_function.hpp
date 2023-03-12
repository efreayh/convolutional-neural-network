#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <string>
#include <functional>
#include "matrix.hpp"
#include "tensor.hpp"

class ActivationFunction {
public:

    /* Constructors */
    ActivationFunction(const std::string& function_name);

    /* Core functionality */
    Tensor apply_function(const Tensor& input_tensor) const;
    Tensor apply_derivative(const Tensor& input_tensor) const;

private:
    /* Functions */
    std::function<Matrix(const Matrix&)> activation_function_;
    std::function<Matrix(const Matrix&)> activation_derivative_;

    /* Activation function type setters */
    void set_sigmoid();
};

#endif