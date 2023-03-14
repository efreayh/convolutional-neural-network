#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <string>
#include "matrix.hpp"

class Tensor {
public:

    /* Constructors */
    Tensor(const int depth, const int rows, const int columns);
    Tensor();
    Tensor(const Matrix& input_data);
    Tensor(const Matrix& input_data, const int depth);
    Tensor(const std::vector<Matrix>& input_data);
    Tensor(const Tensor& other);
    
    /* Accessors */
    int get_num_rows() const;
    int get_num_columns() const;
    int get_depth() const;
    Matrix& operator()(const int index);
    const Matrix& operator()(const int index) const;

    /* Element wise operations applied to each matrix */
    Tensor operator+(const Tensor& other) const;
    Tensor& operator+=(const Tensor& other);
    Tensor operator-(const Tensor& other) const;
    Tensor& operator-=(const Tensor& other);
    Tensor operator*(const Tensor& other) const;
    Tensor element_wise_multiply(const Tensor& other) const;
    Tensor scalar_multiply(const double multiplier) const;
    Tensor transpose() const;
    Tensor max_pool_forward(const int window_size, const int stride) const;
    Tensor max_pool_backward(const Tensor& output, const int window_size, const int stride) const;

    /* Neural network operations */
    Tensor flatten() const;

    /* Other operations */
    Tensor& operator=(const Tensor& other);
    void randomize();
    void reshape(const int depth, const int rows, const int columns);
    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;
    void append_matrix(const Matrix& input_data);

    /* Print operations */
    void print() const;
    void print_dims() const;

private:
    int depth_;
    int rows_;
    int columns_;
    std::vector<Matrix> data_;
};

#endif