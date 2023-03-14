#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <string>

class Matrix {
public:

    /* Constructors */
    Matrix(const int rows, const int columns);
    Matrix(const std::vector<std::vector<double>>& input_matrix);
    Matrix(const Matrix& other);

    /* Accessors */
    int get_num_rows() const;
    int get_num_columns() const;
    double& operator()(const int row, const int column);
    const double& operator()(const int row, const int column) const;
    double get_minimum() const;

    /* Matrix operations */
    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix operator-(const Matrix& other) const;
    Matrix& operator-=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    Matrix element_wise_multiply(const Matrix& other) const;
    Matrix scalar_multiply(const double multiplier) const;
    Matrix transpose() const;

    /* Neural network operations */
    Matrix correlate(const Matrix& filter, const int stride, const std::string& padding_type) const;
    Matrix convolve(const Matrix& filter, const int stride, const std::string& padding_type) const;
    Matrix max_pool_forward(const int window_size, const int stride) const;
    Matrix max_pool_backward(const Matrix& output, const int window_size, const int stride) const;

    /* Other operations */
    Matrix& operator=(const Matrix& other);
    void randomize();
    void reshape(const int rows, const int columns);
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;

    /* Print operations */
    void print() const;
    void print_dims() const;

private:
    int rows_;
    int columns_;
    std::vector<double> data_;
};

#endif