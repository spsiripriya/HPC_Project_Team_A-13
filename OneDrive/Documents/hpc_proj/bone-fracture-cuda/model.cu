#include "model.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

SimpleMLP::SimpleMLP(int in_size, int h_size, int out_size) 
    : input_size(in_size), hidden_size(h_size), output_size(out_size) {
    
    // Initialize weights with He initialization
    std::mt19937 rng(12345);
    float scale = sqrtf(2.0f / input_size);
    std::normal_distribution<float> dist(0.0f, scale);
    
    std::vector<float> W1(hidden_size * input_size);
    std::vector<float> b1(hidden_size, 0.0f);
    std::vector<float> W2(output_size * hidden_size);
    std::vector<float> b2(output_size, 0.0f);
    
    for (auto &w : W1) w = dist(rng);
    for (auto &w : W2) w = dist(rng) * sqrtf(2.0f / hidden_size);
    
    // Allocate and copy weights
    CUDA_CHECK(cudaMalloc(&d_W1, W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, b2.size() * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_W1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate gradients
    CUDA_CHECK(cudaMalloc(&d_gW1, W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb1, b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gW2, W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb2, b2.size() * sizeof(float)));
    
    // Allocate activations
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_drop1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gh1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ga1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gdrop1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gout, output_size * sizeof(float)));
}

SimpleMLP::~SimpleMLP() {
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_gW1); cudaFree(d_gb1); cudaFree(d_gW2); cudaFree(d_gb2);
    cudaFree(d_input); cudaFree(d_h1); cudaFree(d_a1); cudaFree(d_drop1);
    cudaFree(d_mask1); cudaFree(d_output); cudaFree(d_gh1); cudaFree(d_ga1);
    cudaFree(d_gdrop1); cudaFree(d_gout);
}

std::vector<float> SimpleMLP::forward(const std::vector<float>& input, bool training) {
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    int threads = 256;
    
    // Layer 1: FC + ReLU
    fc_forward_kernel<<<(hidden_size + threads - 1) / threads, threads>>>(
        d_input, d_W1, d_b1, d_h1, input_size, hidden_size);
    relu_kernel<<<(hidden_size + threads - 1) / threads, threads>>>(
        d_h1, d_a1, hidden_size);
    
    // Dropout (only during training)
    if (training) {
        unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        dropout_forward_kernel<<<(hidden_size + threads - 1) / threads, threads>>>(
            d_a1, d_drop1, d_mask1, seed, hidden_size, DROPOUT_RATE);
    } else {
        CUDA_CHECK(cudaMemcpy(d_drop1, d_a1, hidden_size * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // Layer 2: FC
    fc_forward_kernel<<<(output_size + threads - 1) / threads, threads>>>(
        d_drop1, d_W2, d_b2, d_output, hidden_size, output_size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> output(output_size);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    return output;
}

void SimpleMLP::backward(const std::vector<float>& grad_output) {
    CUDA_CHECK(cudaMemcpy(d_gout, grad_output.data(), grad_output.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    int threads = 256;
    
    // Layer 2 backward
    int max_size = std::max({output_size * hidden_size, output_size, hidden_size});
    fc_backward_kernel<<<(max_size + threads - 1) / threads, threads>>>(
        d_drop1, d_gout, d_gW2, d_gb2, d_gdrop1, d_W2, hidden_size, output_size);
    
    // Dropout backward
    dropout_backward_kernel<<<(hidden_size + threads - 1) / threads, threads>>>(
        d_gdrop1, d_mask1, d_ga1, hidden_size);
    
    // ReLU backward
    relu_backward_kernel<<<(hidden_size + threads - 1) / threads, threads>>>(
        d_h1, d_ga1, d_gh1, hidden_size);
    
    // Layer 1 backward
    max_size = std::max({hidden_size * input_size, hidden_size, input_size});
    fc_backward_kernel<<<(max_size + threads - 1) / threads, threads>>>(
        d_input, d_gh1, d_gW1, d_gb1, d_input, d_W1, input_size, hidden_size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void SimpleMLP::update(float lr) {
    int threads = 256;
    
    sgd_update_kernel<<<(hidden_size * input_size + threads - 1) / threads, threads>>>(
        d_W1, d_gW1, lr, hidden_size * input_size);
    sgd_update_kernel<<<(hidden_size + threads - 1) / threads, threads>>>(
        d_b1, d_gb1, lr, hidden_size);
    sgd_update_kernel<<<(output_size * hidden_size + threads - 1) / threads, threads>>>(
        d_W2, d_gW2, lr, output_size * hidden_size);
    sgd_update_kernel<<<(output_size + threads - 1) / threads, threads>>>(
        d_b2, d_gb2, lr, output_size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}