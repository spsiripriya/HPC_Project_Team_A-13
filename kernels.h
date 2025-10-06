// kernels.h - Header file for CUDA kernels

#ifndef KERNELS_H
#define KERNELS_H

// Dropout kernels
__global__ void dropout_forward_kernel(const float* input, float* output, float* mask,
                                       unsigned long long seed, int n, float rate);

__global__ void dropout_backward_kernel(const float* grad_out, const float* mask,
                                        float* grad_in, int n);

// Fully connected layer kernels
__global__ void fc_forward_kernel(const float* input, const float* W, const float* b,
                                  float* output, int in_size, int out_size);

__global__ void fc_backward_kernel(const float* input, const float* grad_out,
                                   float* grad_W, float* grad_b, float* grad_in,
                                   const float* W, int in_size, int out_size);

// Activation function kernels
__global__ void relu_kernel(const float* input, float* output, int n);

__global__ void relu_backward_kernel(const float* input, const float* grad_out,
                                     float* grad_in, int n);

// Optimizer kernel
__global__ void sgd_update_kernel(float* params, const float* grads, float lr, int n);

#endif // KERNELS_H
