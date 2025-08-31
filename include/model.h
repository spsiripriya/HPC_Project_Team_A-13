
#ifndef MODEL_H
#define MODEL_H

#include "layers.h"
#include <string>

struct SimpleCNN {
    Conv2D conv1;
    ReLU relu1;
    MaxPool2 pool1;
    Conv2D conv2;
    ReLU relu2;
    MaxPool2 pool2;
    FC fc1;
    FC fc2;

    // intermediate caches
    int c1_h, c1_w, p1_h, p1_w, c2_h, c2_w, p2_h, p2_w;
    std::vector<float> x0, x1, a1, p1v, x2, a2, p2v, flat, fc1_out;

    SimpleCNN();
    std::vector<float> forward(const std::vector<float>& img);
    void backward(const std::vector<float>& logits, int label);
    void step(float lr);
    void save(const std::string &path);
};

#endif

