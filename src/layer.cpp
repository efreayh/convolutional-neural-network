#include "layer.hpp"

/******************************************************
 * Constructors
 *****************************************************/

Layer::Layer(int input_size, int output_size): 
    input_size_(input_size), 
    output_size_(output_size),
    weights_(input_size, output_size), 
    biases_(1, output_size),
    input_(1, input_size),
    output_(1, output_size) {
    
    weights_.randomize();
}

/******************************************************
 * Getters
 *****************************************************/

int Layer::get_input_size() const {
    return input_size_;
}

int Layer::get_output_size() const {
    return output_size_;
}
