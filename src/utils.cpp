
#include "utils.h"
#include <algorithm>
#include <cmath>

std::mt19937 rng(12345);

float randf(float a, float b) {
    std::uniform_real_distribution<float> dist(a, b);
    return dist(rng);
}

std::vector<float> softmax_vec(const std::vector<float>& z) {
    float maxv = *std::max_element(z.begin(), z.end());
    std::vector<float> ex(z.size());
    float sum = 0.0f;
    for (size_t i = 0; i < z.size(); ++i) {
        ex[i] = std::exp(z[i] - maxv);
        sum += ex[i];
    }
    for (size_t i = 0; i < z.size(); ++i) {
        ex[i] /= (sum + 1e-12f);
    }
    return ex;
}

float cross_entropy_loss(const std::vector<float>& probs, int label) {
    float p = std::max(probs[label], 1e-9f);
    return -std::log(p);
}

std::vector<float> dloss_dz(const std::vector<float>& probs, int label) {
    std::vector<float> g(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        g[i] = probs[i];
    }
    g[label] -= 1.0f;
    return g;
}

