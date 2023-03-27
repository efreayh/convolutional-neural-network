#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "mnist_data_set.hpp"

/******************************************************
 * Constructors
 *****************************************************/

MNISTDataSet::MNISTDataSet(const std::string& file_path) {

    std::ifstream file(file_path);
    std::vector<std::vector<double>> one_hot_labels;
    std::vector<std::vector<double>> pixel_values;
    
    /* Get file data */
    if (file.is_open()) {
        std::string line;
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;

            std::getline(ss, token, ',');
            int label = std::stoi(token);

            std::vector<double> one_hot(10, 0.0);
            one_hot[label] = 1.0;
            one_hot_labels.push_back(one_hot);

            std::vector<double> pixels;
            while (std::getline(ss, token, ',')) {
                double pixel_value = std::stod(token);
                pixels.push_back(pixel_value);
            }
            pixel_values.push_back(pixels);
        }
        file.close();
    }

    /* Normalize data */
    for (size_t i = 0; i < pixel_values.size(); ++i) {
        for (size_t j = 0; j < pixel_values[i].size(); ++j) {
            pixel_values[i][j] /= 255.0;
        }
    }

    /* Shuffle the data */
    std::vector<std::vector<double>> data;
    for (size_t i = 0; i < one_hot_labels.size(); ++i) {
        data.push_back(std::vector<double>());
        data.back().insert(data.back().end(), one_hot_labels[i].begin(), one_hot_labels[i].end());
        data.back().insert(data.back().end(), pixel_values[i].begin(), pixel_values[i].end());
    }
    std::random_shuffle(data.begin(), data.end());

    /* Split into train and test sets */
    const double train_ratio = 0.8;
    const size_t train_size = std::floor(data.size() * train_ratio);
    auto train_end = data.begin() + train_size;
    auto train = std::vector<std::vector<double>>(data.begin(), train_end);
    auto test = std::vector<std::vector<double>>(train_end, data.end());

    /* Separate labels and data */
    for (size_t i = 0; i < train.size(); ++i) {
        train_labels_.push_back(std::vector<double>(train[i].begin(), train[i].begin() + 10));
        train_data_.push_back(std::vector<double>(train[i].begin() + 10, train[i].end()));
    }
    for (size_t i = 0; i < test.size(); ++i) {
        test_labels_.push_back(std::vector<double>(test[i].begin(), test[i].begin() + 10));
        test_data_.push_back(std::vector<double>(test[i].begin() + 10, test[i].end()));
    }

    /* Save train and test set sizes */
    train_size_ = train.size();
    test_size_ = test.size();
}

/******************************************************
 * Getters
 *****************************************************/

int MNISTDataSet::get_train_size() const {
    return train_size_;
}

int MNISTDataSet::get_test_size() const {
    return test_size_;
}

Tensor MNISTDataSet::get_train_data(const int position) const {
    if (position < 0 || position >= train_size_) {
        throw std::invalid_argument("MNISTDataSet get_train_data: position out of bounds");
    }
    std::vector<std::vector<double>> result_vec = {train_data_[position]};
    Matrix result_matrix(result_vec);
    Tensor result_tensor(result_matrix);
    result_tensor.reshape(1, 28, 28);
    return result_tensor;
}

Tensor MNISTDataSet::get_train_label(const int position) const {
    if (position < 0 || position >= train_size_) {
        throw std::invalid_argument("MNISTDataSet get_train_label: position out of bounds");
    }
    std::vector<std::vector<double>> result_vec = {train_labels_[position]};
    Matrix result_matrix(result_vec);
    Tensor result_tensor(result_matrix);
    return result_tensor;
}

Tensor MNISTDataSet::get_test_data(const int position) const {
    if (position < 0 || position >= test_size_) {
        throw std::invalid_argument("MNISTDataSet get_test_data: position out of bounds");
    }
    std::vector<std::vector<double>> result_vec = {test_data_[position]};
    Matrix result_matrix(result_vec);
    Tensor result_tensor(result_matrix);
    result_tensor.reshape(1, 28, 28);
    return result_tensor;
}

Tensor MNISTDataSet::get_test_label(const int position) const {
    if (position < 0 || position >= test_size_) {
        throw std::invalid_argument("MNISTDataSet get_test_label: position out of bounds");
    }
    std::vector<std::vector<double>> result_vec = {test_labels_[position]};
    Matrix result_matrix(result_vec);
    Tensor result_tensor(result_matrix);
    return result_tensor;
}
