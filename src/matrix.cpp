#include <iostream>
#include <stdexcept>
#include <random>
#include "matrix.hpp"
#include "utility.hpp"

/******************************************************
 * Constructors
 *****************************************************/

Matrix::Matrix(int rows, int columns) {
    if (rows <= 0 || columns <= 0) {
        throw std::invalid_argument("Matrix constructor: dimensions must be greater than 0");
    }

    rows_ = rows;
    columns_ = columns;
    data_.resize(rows_ * columns_, 0.0);
}

Matrix::Matrix(const std::vector<std::vector<double>>& input_matrix) {
    if (input_matrix.empty() || input_matrix[0].empty()) {
        throw std::invalid_argument("Matrix constructor: input matrix cannot be empty");
    }

    rows_ = input_matrix.size();
    columns_ = input_matrix[0].size();
    data_.reserve(rows_ * columns_);
    for (int i = 0; i < rows_; ++i) {
        data_.insert(data_.end(), input_matrix[i].begin(), input_matrix[i].end());
    }
}

Matrix::Matrix(const Matrix& other) {
    rows_ = other.rows_;
    columns_ = other.columns_;
    data_ = other.data_;
}

/******************************************************
 * Accessors
 *****************************************************/

int Matrix::get_num_rows() const {
    return rows_;
}

int Matrix::get_num_columns() const {
    return columns_;
}

double& Matrix::operator()(const int row, const int column) {
    if (row < 0 || row >= rows_ || column < 0 || column >= columns_) {
        throw std::invalid_argument("Matrix accessor: coordinates out of bounds");
    }

    return data_[row * columns_ + column];
}

const double& Matrix::operator()(const int row, const int column) const {
    if (row < 0 || row >= rows_ || column < 0 || column >= columns_) {
        throw std::invalid_argument("Matrix accessor: coordinates out of bounds");
    }
    
    return data_[row * columns_ + column];
}

/******************************************************
 * Matrix operations
 *****************************************************/

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || columns_ != other.columns_) {
        throw std::invalid_argument("Matrix addition: dimensions do not match");
    }

    Matrix result(rows_, columns_);
    for (int i = 0; i < rows_ * columns_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || columns_ != other.columns_) {
        throw std::invalid_argument("Matrix addition assignment: dimensions do not match");
    }

    for (int i = 0; i < rows_ * columns_; ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || columns_ != other.columns_) {
        throw std::invalid_argument("Matrix subtraction: dimensions do not match");
    }

    Matrix result(rows_, columns_);
    for (int i = 0; i < rows_ * columns_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || columns_ != other.columns_) {
        throw std::invalid_argument("Matrix subtraction assignment: dimensions do not match");
    }

    for (int i = 0; i < rows_ * columns_; ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (columns_ != other.rows_) {
        throw std::invalid_argument("Matrix multiplication: dimensions are incompatible");
    }

    Matrix result(rows_, other.columns_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < other.columns_; ++j) {
            double sum = 0.0;
            for (int k = 0; k < columns_; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::element_wise_multiply(const Matrix& other) const {
    if (rows_ != other.rows_ || columns_ != other.columns_) {
        throw std::invalid_argument("Matrix element wise multiply: dimensions do not match");
    }

    Matrix result(rows_, columns_);
    for (int i = 0; i < rows_ * columns_; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Matrix Matrix::scalar_multiply(const double multiplier) const {
    Matrix result(rows_, columns_);
    for (int i = 0; i < rows_ * columns_; ++i) {
        result.data_[i] = data_[i] * multiplier;
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(columns_, rows_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < columns_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

/******************************************************
 * Neural network operations
 *****************************************************/

Matrix Matrix::convolve(const Matrix& filter, int stride, std::string padding_type) const {
    if (utility::compare_ignore_case(padding_type, "full")) {
        throw std::logic_error("Matrix convolve: unimplemented");
    }
    else if (utility::compare_ignore_case(padding_type, "same")) {
        throw std::logic_error("Matrix convolve: unimplemented");
    }
    else if (utility::compare_ignore_case(padding_type, "valid")) {
        throw std::logic_error("Matrix convolve: unimplemented");
    }
    else {
        throw std::invalid_argument("Matrix convolve: invalid padding_type");
    }
}

Matrix Matrix::max_pool(int window_size, int stride) const {
    throw std::logic_error("Matrix max_pool: unimplemented");
}

/******************************************************
 * Other operations
 *****************************************************/

Matrix& Matrix::operator=(const Matrix& other) {
    if (&other == this) {
        return *this;
    }

    rows_ = other.rows_;
    columns_ = other.columns_;
    data_ = other.data_;

    return *this;
}

void Matrix::randomize() {
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 1);

    for (int i = 0; i < rows_ * columns_; ++i) {
        data_[i] = dist(generator);
    }
}

bool Matrix::operator==(const Matrix& other) const {
    if (rows_ != other.rows_ || columns_ != other.columns_) {
        return false;
    }

    for (int i = 0; i < rows_ * columns_; ++i) {
        if (data_[i] != other.data_[i]) {
            return false;
        }
    }

    return true;
}

bool Matrix::operator!=(const Matrix& other) const {
    if (rows_ != other.rows_ || columns_ != other.columns_) {
        return true;
    }

    for (int i = 0; i < rows_ * columns_; ++i) {
        if (data_[i] != other.data_[i]) {
            return true;
        }
    }

    return false;
}

/******************************************************
 * Print operations
 *****************************************************/

void Matrix::print() const {
    for (int i = 0; i < rows_ * columns_; ++i) {
        std::cout << data_[i] << " ";
        if ((i + 1) % columns_ == 0) {
            std::cout << std::endl;
        }
    }
}

void Matrix::print_dims() const {
    std::cout << "Rows: " << rows_ << " Cols: " << columns_ << std::endl;
}
