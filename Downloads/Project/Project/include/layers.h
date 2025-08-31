
#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <iostream>

// Conv2D Layer
struct Conv2D {
    int in_c, out_c, k;
    int in_h, in_w;
    std::vector<float> kernels;
    std::vector<float> grad_kernels;

    Conv2D();
    Conv2D(int in_channels, int out_channels, int kernel_size);
    
    std::vector<float> forward(const std::vector<float>& input, int H, int W);
    std::vector<float> backward(const std::vector<float>& input, int H, int W,
                                const std::vector<float>& d_out);
    void step(float lr);
    void save(std::ostream &os);
};

// ReLU Activation
struct ReLU {
    std::vector<float> last_in;
    
    std::vector<float> forward(const std::vector<float>& x);
    std::vector<float> backward(const std::vector<float>& grad_out);
};

// MaxPool2x2
struct MaxPool2 {
    int in_c, in_h, in_w;
    int out_h, out_w;
    std::vector<int> max_idx;
    
    MaxPool2();
    std::vector<float> forward(const std::vector<float>& x, int C, int H, int W);
    std::vector<float> backward(const std::vector<float>& grad_out);
};

// Fully Connected Layer
struct FC {
    int in_size, out_size;
    std::vector<float> W;
    std::vector<float> b;
    std::vector<float> last_in;
    std::vector<float> grad_W;
    std::vector<float> grad_b;

    FC();
    FC(int in_s, int out_s);
    
    std::vector<float> forward(const std::vector<float>& in);
    std::vector<float> backward(const std::vector<float>& grad_out);
    void step(float lr);
    void save(std::ostream &os);
};

#endif

