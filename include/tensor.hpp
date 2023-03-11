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
    
    /* Accessors */
    int get_num_rows() const;
    int get_num_columns() const;
    int get_depth() const;
    Matrix& operator()(const int index);
    const Matrix& operator()(const int index) const;

    /* Tensor operations */
    void append_matrix(const Matrix& input_data);

    /* Neural network operations */
    Tensor convolve(const Tensor& filters, const int stride, const std::string& padding_type) const;
    Tensor max_pool(const int window_size, const int stride) const;
    Tensor flatten() const;

    /* Print operations */
    void print_dims() const;

private:
    int depth_;
    int rows_;
    int columns_;
    std::vector<Matrix> data_;
};

#endif