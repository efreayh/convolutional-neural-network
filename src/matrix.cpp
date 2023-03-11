#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>
#include "matrix.hpp"
#include "utility.hpp"

/******************************************************
 * Constructors
 *****************************************************/

Matrix::Matrix(const int rows, const int columns) {
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

double Matrix::get_minimum() const {
    double min = INFINITY;
    for (int i = 0; i < rows_ * columns_; ++i) {
        if (data_[i] < min) {
            min = data_[i];
        }
    }
    return min;
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

Matrix Matrix::correlate(const Matrix& filter, const int stride, const std::string& padding_type) const {
    if (stride > filter.rows_ || stride > filter.columns_) {
    throw std::invalid_argument("Matrix correlate/convolve: stride must be less than or equal to filter size");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Matrix correlate/convolve: stride must be greater than 0");
    }
    if (filter.rows_ > rows_ || filter.columns_ > columns_) {
        throw std::invalid_argument("Matrix correlate/convolve: filter size must be less or equal to matrix dimensions");
    }


    if (utility::compare_ignore_case(padding_type, "full")) {

        if (filter.rows_ < 2 ||
            filter.columns_ < 2 ||
            (rows_ + filter.rows_ - 2) % stride != 0 ||
            (columns_ + filter.columns_ - 2) % stride != 0) {
            throw std::invalid_argument("Matrix correlate/convolve: Full padding is not possible for given filter and stride sizes");
        }
        
        int result_rows = (rows_ + filter.rows_ - 2) / stride + 1;
        int result_columns = (columns_ + filter.columns_ - 2) / stride + 1;
        Matrix result(result_rows, result_columns);

        int padding_rows = (result_rows - 1) * stride + filter.rows_ - rows_;
        int padding_columns = (result_columns - 1) * stride + filter.columns_ - columns_;

        int padding_top = padding_rows / 2 + padding_rows % 2;
        int padding_left = padding_columns / 2 + padding_columns % 2;

        for (int i = 0; i < rows_ + padding_rows - filter.rows_ + 1; i += stride) {
            for (int j = 0; j < columns_ + padding_columns - filter.columns_ + 1; j += stride) {
                double sum = 0.0;

                for (int k = 0; k < filter.rows_; ++k) {
                    for (int l = 0; l < filter.columns_; ++l) {
                        if (i + k >= padding_top &&
                            i + k < padding_top + rows_ &&
                            j + l >= padding_left &&
                            j + l < padding_left + columns_) {

                            sum += (*this)(i + k - padding_top, j + l - padding_left) * filter(k, l);
                        }
                    }
                }

                result(i / stride, j / stride) = sum;
            }
        }

        return result;
    }
    else if (utility::compare_ignore_case(padding_type, "same")) {

        int result_rows = rows_;
        int result_columns = columns_;
        Matrix result(result_rows, result_columns);

        int padding_rows = (result_rows - 1) * stride + filter.rows_ - rows_;
        int padding_columns = (result_columns - 1) * stride + filter.columns_ - columns_;

        int padding_top = padding_rows / 2 + padding_rows % 2;
        int padding_left = padding_columns / 2 + padding_columns % 2;

        for (int i = 0; i < rows_ + padding_rows - filter.rows_ + 1; i += stride) {
            for (int j = 0; j < columns_ + padding_columns - filter.columns_ + 1; j += stride) {
                double sum = 0.0;

                for (int k = 0; k < filter.rows_; ++k) {
                    for (int l = 0; l < filter.columns_; ++l) {
                        if (i + k >= padding_top &&
                            i + k < padding_top + rows_ &&
                            j + l >= padding_left &&
                            j + l < padding_left + columns_) {

                            sum += (*this)(i + k - padding_top, j + l - padding_left) * filter(k, l);
                        }
                    }
                }

                result(i / stride, j / stride) = sum;
            }
        }

        return result;
    }
    else if (utility::compare_ignore_case(padding_type, "valid")) {

        if ((rows_ - filter.rows_) % stride != 0 ||
            (columns_ - filter.columns_) % stride != 0 ||
            (rows_ == filter.rows_ && stride < rows_) ||
            (columns_ == filter.columns_ && stride < columns_)) {
            throw std::invalid_argument("Matrix correlate/convolve: Valid padding is not possible for given filter and stride sizes");
        }

        int result_rows = (rows_ - filter.rows_) / stride + 1;
        int result_columns = (columns_ - filter.columns_) / stride + 1;
        Matrix result(result_rows, result_columns);

        for (int i = 0; i < rows_ - filter.rows_ + 1; i += stride) {
            for (int j = 0; j < columns_ - filter.columns_ + 1; j += stride) {
                double sum = 0.0;

                for (int k = 0; k < filter.rows_; ++k) {
                    for (int l = 0; l < filter.columns_; ++l) {
                        sum += (*this)(i + k, j + l) * filter(k, l);
                    }
                }

                result(i / stride, j / stride) = sum;
            }
        }

        return result;
    }
    else {
        throw std::invalid_argument("Matrix correlate/convolve: invalid padding_type");
    }
}

Matrix Matrix::convolve(const Matrix& filter, const int stride, const std::string& padding_type) const {
    
    /* Rotate filter 180 degrees */
    Matrix new_filter(filter.rows_, filter.columns_);

    for (int i = 0; i < filter.rows_; ++i) {
        for (int j = 0; j < filter.columns_; ++j) {
            new_filter(i, j) = filter(filter.rows_ - 1 - i, filter.columns_ - 1 - j);
        }
    }

    /* Run correlate with rotated filter */
    return correlate(new_filter, stride, padding_type);
}

Matrix Matrix::max_pool(const int window_size, const int stride) const {
    if (stride > window_size) {
        throw std::invalid_argument("Matrix max_pool: stride must be less than or equal to window_size");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Matrix max_pool: stride must be greater than 0");
    }
    if (window_size > rows_ || window_size > columns_) {
        throw std::invalid_argument("Matrix max_pool: window_size must be less or equal to matrix dimensions");
    }
    if (get_minimum() < 0.0) {
        throw std::logic_error("Matrix max_pool: matrix contains negative numbers, cannot add zero padding");
    }

    int result_rows = ((rows_ - window_size) + stride - 1) / stride + 1;
    int result_columns = ((columns_ - window_size) + stride - 1) / stride + 1;
    Matrix result(result_rows, result_columns);

    int padding_rows = (result_rows - 1) * stride + window_size - rows_;
    int padding_columns = (result_columns - 1) * stride + window_size - columns_;

    int padding_top = padding_rows / 2 + padding_rows % 2;
    int padding_left = padding_columns / 2 + padding_columns % 2;

    for (int i = 0; i < rows_ + padding_rows - window_size + 1; i += stride) {
        for (int j = 0; j < columns_ + padding_columns - window_size + 1; j += stride) {
            double max = 0.0;

            for (int k = 0; k < window_size; ++k) {
                for (int l = 0; l < window_size; ++l) {
                    if (i + k >= padding_top &&
                        i + k < padding_top + rows_ &&
                        j + l >= padding_left &&
                        j + l < padding_left + columns_ &&
                        (*this)(i + k - padding_top, j + l - padding_left) > max) {

                        max = (*this)(i + k - padding_top, j + l - padding_left);
                    }
                }
            }

            result(i / stride, j / stride) = max;
        }
    }

    return result;
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
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> dist(0, 1);

    for (int i = 0; i < rows_ * columns_; ++i) {
        data_[i] = dist(generator);
    }
}

void Matrix::resize(const int rows, const int columns) {
    if (rows < 1 || columns < 1) {
        throw std::invalid_argument("Matrix resize: new dimensions cannot be zero or less");
    }
    if (rows * columns != rows_ * columns_) {
        throw std::invalid_argument("Matrix resize: new dimensions invalid for current matrix data");
    }

    rows_ = rows;
    columns_ = columns;
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
