#include <stdexcept>
#include <iostream>
#include "tensor.hpp"

/******************************************************
 * Constructors
 *****************************************************/

Tensor::Tensor(const int depth, const int rows, const int columns):
    depth_(depth),
    rows_(rows),
    columns_(columns) {
    
    for (int i = 0; i < depth_; ++i) {
        Matrix matrix(rows, columns);
        matrix.randomize();
        data_.push_back(matrix);
    }
}

Tensor::Tensor(): Tensor(0, 0, 0) {}

Tensor::Tensor(const Matrix& input_data):
    Tensor(input_data, 1) {}

Tensor::Tensor(const Matrix& input_data, const int depth) {
    if (depth <= 0) {
        throw std::invalid_argument("Tensor constructor: depth must at least 1 when input_data is supplied");
    }

    depth_ = depth;
    rows_ = input_data.get_num_rows();
    columns_ = input_data.get_num_columns();

    for (int i = 0; i < depth_; ++i) {
        data_.push_back(input_data);
    }
}

Tensor::Tensor(const std::vector<Matrix>& input_data) {
    if (input_data.empty()) {
        throw std::invalid_argument("Tensor constructor: input_data was empty");
    }

    depth_ = input_data.size();
    rows_ = input_data[0].get_num_rows();
    columns_ = input_data[0].get_num_columns();

    for (int i = 0; i < depth_; ++i) {
        if (input_data[i].get_num_rows() == rows_ && input_data[i].get_num_columns() == columns_) {
            data_.push_back(input_data[i]);
        }
        else {
            throw std::invalid_argument("Tensor constructor: input_data matrices must all have equal dimensions");
        }
    }
}

/******************************************************
 * Accessors
 *****************************************************/

int Tensor::get_num_rows() const {
    return rows_;
}

int Tensor::get_num_columns() const {
    return columns_;
}

int Tensor::get_depth() const {
    return depth_;
}

Matrix& Tensor::operator()(const int index) {
    if (index < 0 || index >= depth_) {
        throw std::invalid_argument("Tensor get_matrix: index out of bounds for tensor depth");
    }

    return data_[index];
}

const Matrix& Tensor::operator()(const int index) const {
    if (index < 0 || index >= depth_) {
        throw std::invalid_argument("Tensor get_matrix: index out of bounds for tensor depth");
    }

    return data_[index];
}

/******************************************************
 * Tensor operations
 *****************************************************/

void Tensor::append_matrix(const Matrix& input_data) {
    if (depth_ <= 0) {
        depth_ = 1;
        rows_ = input_data.get_num_rows();
        columns_ = input_data.get_num_columns();
        data_.push_back(input_data);
    }
    else if (input_data.get_num_rows() == rows_ && input_data.get_num_columns() == columns_) {
        ++depth_;
        data_.push_back(input_data);
    }
    else {
        throw std::invalid_argument("Tensor append_matrix: input matrix must match tensor dimensions");
    }
}

/******************************************************
 * Neural network operations
 *****************************************************/

Tensor Tensor::convolve(const Tensor& filters, const int stride, const std::string& padding_type) const {
    if (depth_ != filters.depth_) {
        throw std::invalid_argument("Tensor convolve: depth of filters does not match depth of tensor");
    }

    Tensor result;
    for (int i = 0; i < depth_; ++i) {
        result.append_matrix(data_[i].convolve(filters.data_[i], stride, padding_type));
    }
    return result;
}

Tensor Tensor::max_pool(const int window_size, const int stride) const {
    Tensor result;
    for (int i = 0; i < depth_; ++i) {
        result.append_matrix(data_[i].max_pool(window_size, stride));
    }
    return result;
}

Tensor Tensor::flatten() const {
    std::vector<double> vector1d;

    for (int i = 0; i < depth_; ++i) {
        for (int j = 0; j < rows_; ++j) {
            for (int k = 0; k < columns_; ++k) {
                vector1d.push_back(data_[i](j, k));
            }
        }
    }

    std::vector<std::vector<double>> vector2d;
    vector2d.push_back(vector1d);
    Matrix matrix(vector2d);
    matrix = matrix.transpose();
    Tensor result(matrix);
    return result;
}

/******************************************************
 * Print operations
 *****************************************************/

void Tensor::print_dims() const {
    std::cout << "Depth: " << depth_ << " Rows: " << rows_ << " Cols: " << columns_ << std::endl;
}

