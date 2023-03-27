#ifndef MNIST_DATA_SET_HPP
#define MNIST_DATA_SET_HPP

#include <string>
#include <vector>
#include "tensor.hpp"

class MNISTDataSet {
public:

    /* Constructors */
    MNISTDataSet(const std::string& file_path);

    /* Getters */
    int get_train_size() const;
    int get_test_size() const;
    Tensor get_train_data(const int position) const;
    Tensor get_train_label(const int position) const;
    Tensor get_test_data(const int position) const;
    Tensor get_test_label(const int position) const;

private:
    int train_size_;
    int test_size_;
    std::vector<std::vector<double>> train_data_;
    std::vector<std::vector<double>> train_labels_;
    std::vector<std::vector<double>> test_data_;
    std::vector<std::vector<double>> test_labels_;
};

#endif