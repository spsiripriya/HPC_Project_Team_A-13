
#include "model.h"
#include "utils.h"
#include <fstream>
#include <algorithm>

SimpleCNN::SimpleCNN()
    : conv1(1, 8, 3), conv2(8, 16, 3), fc1(), fc2(),
      c1_h(0), c1_w(0), p1_h(0), p1_w(0), c2_h(0), c2_w(0), p2_h(0), p2_w(0) {
}

std::vector<float> SimpleCNN::forward(const std::vector<float>& img) {
    // input: img flattened [H*W]
    x0.assign(img.begin(), img.end());

    // conv1
    x1 = conv1.forward(x0, 64, 64); // IMG_SIZE = 64
    c1_h = 64 - conv1.k + 1;
    c1_w = 64 - conv1.k + 1;

    // relu1
    a1 = relu1.forward(x1);

    // pool1
    p1v = pool1.forward(a1, conv1.out_c, c1_h, c1_w);
    p1_h = c1_h / 2; 
    p1_w = c1_w / 2;

    // conv2
    x2 = conv2.forward(p1v, p1_h, p1_w);
    c2_h = p1_h - conv2.k + 1;
    c2_w = p1_w - conv2.k + 1;

    // relu2
    a2 = relu2.forward(x2);

    // pool2
    p2v = pool2.forward(a2, conv2.out_c, c2_h, c2_w);
    p2_h = c2_h / 2; 
    p2_w = c2_w / 2;

    // flatten
    int tot = conv2.out_c * p2_h * p2_w;
    flat.resize(tot);
    for (int i = 0; i < tot; ++i) {
        flat[i] = p2v[i];
    }

    // init fc1 if needed
    if (fc1.in_size == 0) {
        fc1 = FC((int)flat.size(), 32);
        fc2 = FC(32, 2);
    }

    fc1_out = fc1.forward(flat);
    // apply relu to fc1_out for fc2 input
    std::vector<float> fc1_act(fc1_out.size());
    for (size_t i = 0; i < fc1_out.size(); ++i) {
        fc1_act[i] = std::max(0.0f, fc1_out[i]);
    }

    std::vector<float> logits = fc2.forward(fc1_act);
    return logits;
}

void SimpleCNN::backward(const std::vector<float>& logits, int label) {
    std::vector<float> probs = softmax_vec(logits);
    std::vector<float> dlog = dloss_dz(probs, label);

    // fc2 backward
    std::vector<float> grad_fc1_act = fc2.backward(dlog);

    // grad through relu on fc1_out
    std::vector<float> grad_fc1_out(grad_fc1_act.size());
    for (size_t i = 0; i < grad_fc1_out.size(); ++i) {
        grad_fc1_out[i] = (fc1_out[i] > 0.0f) ? grad_fc1_act[i] : 0.0f;
    }

    // fc1 backward
    std::vector<float> grad_flat = fc1.backward(grad_fc1_out);

    // reshape grad_flat -> p2 shape
    std::vector<float> grad_p2 = grad_flat;

    // pool2 backward
    std::vector<float> grad_a2 = pool2.backward(grad_p2);

    // relu2 backward
    std::vector<float> grad_x2 = relu2.backward(grad_a2);

    // conv2 backward
    std::vector<float> grad_p1 = conv2.backward(p1v, p1_h, p1_w, grad_x2);

    // pool1 backward
    std::vector<float> grad_a1 = pool1.backward(grad_p1);

    // relu1 backward
    std::vector<float> grad_x1 = relu1.backward(grad_a1);

    // conv1 backward
    std::vector<float> grad_input = conv1.backward(x0, 64, 64, grad_x1);
}

void SimpleCNN::step(float lr) {
    conv1.step(lr);
    conv2.step(lr);
    fc1.step(lr);
    fc2.step(lr);
}

void SimpleCNN::save(const std::string &path) {
    std::ofstream os(path);
    conv1.save(os);
    conv2.save(os);
    fc1.save(os);
    fc2.save(os);
    os.close();
}

