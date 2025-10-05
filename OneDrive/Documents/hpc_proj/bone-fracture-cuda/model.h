// model.h - Header file for neural network model
// Contributor: Person 3

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

const float DROPOUT_RATE = 0.3f;

class SimpleMLP {
public:
    int input_size, hidden_size, output_size;
    
    // Model parameters (device pointers)
    float *d_W1, *d_b1, *d_W2, *d_b2;
    
    // Gradients (device pointers)
    float *d_gW1, *d_gb1, *d_gW2, *d_gb2;
    
    // Activations and intermediate values (device pointers)
    float *d_input, *d_h1, *d_a1, *d_drop1, *d_mask1, *d_output;
    float *d_gh1, *d_ga1, *d_gdrop1, *d_gout;
    
    // Constructor and destructor
    SimpleMLP(int in_size, int h_size, int out_size);
    ~SimpleMLP();
    
    // Main methods
    std::vector<float> forward(const std::vector<float>& input, bool training = true);
    void backward(const std::vector<float>& grad_output);
    void update(float lr);
};

#endif // MODEL_H