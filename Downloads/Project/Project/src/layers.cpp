
#include "layers.h"
#include "utils.h"
#include <algorithm>

// Conv2D Implementation
Conv2D::Conv2D() : in_c(0), out_c(0), k(0), in_h(0), in_w(0) {}

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size)
    : in_c(in_channels), out_c(out_channels), k(kernel_size), in_h(0), in_w(0) {
    int K = out_c * in_c * k * k;
    kernels.resize(K);
    grad_kernels.resize(K);
    for (int i = 0; i < K; i++) {
        kernels[i] = randf();
    }
}

std::vector<float> Conv2D::forward(const std::vector<float>& input, int H, int W) {
    in_h = H; 
    in_w = W;
    int OH = H - k + 1;
    int OW = W - k + 1;
    std::vector<float> out(out_c * OH * OW, 0.0f);

    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < k; ++kh) {
                        for (int kw = 0; kw < k; ++kw) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            float val = input[idx3(ic, ih, iw, H, W)];
                            int kidx = (((oc * in_c + ic) * k + kh) * k + kw);
                            sum += val * kernels[kidx];
                        }
                    }
                }
                out[(oc * OH + oh) * OW + ow] = sum;
            }
        }
    }
    return out;
}

std::vector<float> Conv2D::backward(const std::vector<float>& input, int H, int W,
                                   const std::vector<float>& d_out) {
  
}

void Conv2D::step(float lr) {
    for (size_t i = 0; i < kernels.size(); ++i) {
        kernels[i] -= lr * grad_kernels[i];
    }
}

void Conv2D::save(std::ostream &os) {
    for (auto &v : kernels) {
        os << v << " ";
    }
    os << "\n";
}

// ReLU Implementation
std::vector<float> ReLU::forward(const std::vector<float>& x) {
    last_in = x;
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] > 0 ? x[i] : 0.0f;
    }
    return y;
}

std::vector<float> ReLU::backward(const std::vector<float>& grad_out) {
  
}

// MaxPool2 Implementation
MaxPool2::MaxPool2() : in_c(0), in_h(0), in_w(0), out_h(0), out_w(0) {}

std::vector<float> MaxPool2::forward(const std::vector<float>& x, int C, int H, int W) {
    in_c = C; 
    in_h = H; 
    in_w = W;
    out_h = H / 2; 
    out_w = W / 2;
    std::vector<float> out(C * out_h * out_w, 0.0f);
    max_idx.assign(C * out_h * out_w, -1);

    for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float best = -1e9f;
                int best_i = -1;
                for (int kh = 0; kh < 2; ++kh) {
                    for (int kw = 0; kw < 2; ++kw) {
                        int ih = oh * 2 + kh;
                        int iw = ow * 2 + kw;
                        int id = idx3(c, ih, iw, H, W);
                        float v = x[id];
                        if (v > best) {
                            best = v; 
                            best_i = id;
                        }
                    }
                }
                int out_idx = (c * out_h + oh) * out_w + ow;
                out[out_idx] = best;
                max_idx[out_idx] = best_i;
            }
        }
    }
    return out;
}

std::vector<float> MaxPool2::backward(const std::vector<float>& grad_out) {
   
}

// FC Implementation
FC::FC() : in_size(0), out_size(0) {}

FC::FC(int in_s, int out_s) : in_size(in_s), out_size(out_s) {
    W.resize(out_size * in_size);
    grad_W.resize(out_size * in_size);
    b.resize(out_size);
    grad_b.resize(out_size);
    for (auto &v : W) {
        v = randf();
    }
    for (auto &v : b) {
        v = 0.0f;
    }
}

std::vector<float> FC::forward(const std::vector<float>& in) {
    last_in = in;
    std::vector<float> out(out_size, 0.0f);
    for (int o = 0; o < out_size; ++o) {
        float s = b[o];
        for (int i = 0; i < in_size; ++i) {
            s += W[o * in_size + i] * in[i];
        }
        out[o] = s;
    }
    return out;
}

std::vector<float> FC::backward(const std::vector<float>& grad_out) {

}

void FC::step(float lr) {
    for (size_t i = 0; i < W.size(); ++i) {
        W[i] -= lr * grad_W[i];
    }
    for (int o = 0; o < out_size; ++o) {
        b[o] -= lr * grad_b[o];
    }
}

void FC::save(std::ostream &os) {
    for (auto &v : W) {
        os << v << " ";
    }
    os << "\n";
    for (auto &v : b) {
        os << v << " ";
    }
    os << "\n";
}
