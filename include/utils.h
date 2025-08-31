#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>

// Global RNG
extern std::mt19937 rng;

// Helper functions
inline int idx3(int c, int h, int w, int H, int W) { 
    return (c * H + h) * W + w; 
}

float randf(float a = -0.08f, float b = 0.08f);

// Softmax and loss functions
std::vector<float> softmax_vec(const std::vector<float>& z);
float cross_entropy_loss(const std::vector<float>& probs, int label);
std::vector<float> dloss_dz(const std::vector<float>& probs, int label);

#endif
