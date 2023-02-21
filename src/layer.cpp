#include <functional>
#include "layer.hpp"

/******************************************************
 * Constructors
 *****************************************************/

Layer::Layer(int input_size, int output_size, std::function<double(double)> function, std::function<double(double)> derivative): 
    input_size_(input_size), 
    output_size_(output_size),
    weights_(output_size, input_size), 
    biases_(output_size, 1),
    input_(input_size, 1),
    output_(output_size, 1),
    activation_function_(function),
    activation_derivative_(derivative) {
    
    weights_.randomize();
    biases_.randomize();
}

Layer::Layer(int input_size, int output_size): 
    Layer(input_size, output_size, [](double x) { return x; }, [](double) { return 1; }) {}

/******************************************************
 * Getters
 *****************************************************/

int Layer::get_input_size() const {
    return input_size_;
}

int Layer::get_output_size() const {
    return output_size_;
}

/******************************************************
 * Setters
 *****************************************************/

void Layer::set_activation_function(std::function<double(double)> function) {
    activation_function_ = function;
}

void Layer::set_activation_derivative(std::function<double(double)> derivative) {
    activation_derivative_ = derivative;
}
