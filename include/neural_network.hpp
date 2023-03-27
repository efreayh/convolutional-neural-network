#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <memory>
#include "tensor.hpp"
#include "layer.hpp"

class NeuralNetwork {
public:

    /* Constructors */
    NeuralNetwork();

    /* Setters */
    void add_layer(std::unique_ptr<Layer> layer);

    /* Operations */
    void train(const Tensor& input, const Tensor& expected_output);
    Tensor predict(const Tensor& input);

private:
    int num_layers_;
    std::vector<std::unique_ptr<Layer>> layers_;
};

#endif