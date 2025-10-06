// kernels.cu - CUDA kernel implementations
// This module contains all CUDA kernel functions for forward and backward operations

#include "kernels.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Dropout kernels
__global__ void dropout_forward_kernel(const float* input, float* output, float* mask,
                                       unsigned long long seed, int n, float rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);
        mask[idx] = (rand_val > rate) ? 1.0f / (1.0f - rate) : 0.0f;
        output[idx] = input[idx] * mask[idx];
    }
}

__global__ void dropout_backward_kernel(const float* grad_out, const float* mask,
                                        float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = grad_out[idx] * mask[idx];
    }
}

// Fully connected layer kernels
__global__ void fc_forward_kernel(const float* input, const float* W, const float* b,
                                  float* output, int in_size, int out_size) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < out_size) {
        float sum = b[o];
        for (int i = 0; i < in_size; ++i) {
            sum += W[o * in_size + i] * input[i];
        }
        output[o] = sum;
    }
}

__global__ void fc_backward_kernel(const float* input, const float* grad_out,
                                   float* grad_W, float* grad_b, float* grad_in,
                                   const float* W, int in_size, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < out_size * in_size) {
        int i = idx % in_size;
        int o = idx / in_size;
        grad_W[idx] = grad_out[o] * input[i];
    }
    
    if (idx < out_size) {
        grad_b[idx] = grad_out[idx];
    }
    
    if (idx < in_size) {
        float sum = 0.0f;
        for (int o = 0; o < out_size; ++o) {
            sum += W[o * in_size + idx] * grad_out[o];
        }
        grad_in[idx] = sum;
    }
}

// Activation function kernels
__global__ void relu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void relu_backward_kernel(const float* input, const float* grad_out,
                                     float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = (input[idx] > 0.0f) ? grad_out[idx] : 0.0f;
    }
}

// Optimizer kernel
__global__ void sgd_update_kernel(float* params, const float* grads, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        params[idx] -= lr * grads[idx];
    }
}
