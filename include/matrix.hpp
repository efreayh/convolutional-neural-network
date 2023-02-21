#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <functional>

class Matrix {
public:

    /* Constructors */
    Matrix(int rows, int columns, int padding);
    Matrix(int rows, int columns);
    Matrix(std::vector<std::vector<double>> const &input_matrix, int padding);
    Matrix(std::vector<std::vector<double>> const &input_matrix);

    /* Accessors */
    int get_num_rows() const;
    int get_num_columns() const;
    int get_padding() const;
    double& operator()(const int row, const int column);
    const double& operator()(const int row, const int column) const;

    /* Matrix operations */
    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix operator-(const Matrix& other) const;
    Matrix& operator-=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    Matrix element_wise_multiply(const Matrix& other) const;
    Matrix scalar_multiply(const double multiplier) const;
    Matrix transpose() const;

    /* Other operations */
    Matrix apply_function(const std::function<double(double)>& function) const;
    void randomize();
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;

    /* Print operations */
    void print() const;

private:
    int rows_;
    int columns_;
    int padding_;
    std::vector<double> data_;
};

#endif