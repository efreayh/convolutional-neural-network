#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <string>
#include <functional>
#include "matrix.hpp"

class ActivationFunction {
public:

    /* Constructors */
    ActivationFunction(const std::string& function_name);

    /* Core functionality */
    Matrix apply_function(const Matrix& input_matrix) const;
    Matrix apply_derivative(const Matrix& input_matrix) const;

private:
    /* Functions */
    std::function<Matrix(const Matrix&)> activation_function_;
    std::function<Matrix(const Matrix&)> activation_derivative_;

    /* Activation function type setters */
    void set_sigmoid();

    /* Helper functions */
    bool compare_ignore_case(const std::string& s1, const std::string& s2);
};

#endif