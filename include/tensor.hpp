#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <string>
#include "matrix.hpp"

class Tensor {
public:

    /* Constructors */
    Tensor(Matrix& input_data);
    Tensor(std::vector<Matrix>& input_data);
    Tensor(int rows, int columns, int depth);
    
    /* Accessors */
    int get_num_rows() const;
    int get_num_columns() const;
    int get_depth() const;
    Matrix get_matrix_if_single_dim() const;

    /* Tensor operations */
    Tensor convolve(const Tensor& filters, int stride, std::string padding_type) const;
    Tensor max_pool(int window_size, int stride) const;
    Tensor flatten() const;

    /* Print operations */
    void print_dims() const;

private:
    int rows_;
    int columns_;
    int depth_;
    std::vector<Matrix> data_;
};

#endif