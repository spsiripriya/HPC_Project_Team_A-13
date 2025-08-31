
#include "training.h"
#include "utils.h"
#include <algorithm>

float evaluate(SimpleCNN &model, const std::vector<Sample>& val) {
    if (val.empty()) return 0.0f;
    int correct = 0;
    for (size_t i = 0; i < val.size(); ++i) {
        auto logits = model.forward(val[i].img);
        auto probs = softmax_vec(logits);
        int pred = std::max_element(probs.begin(), probs.end()) - probs.begin();
        if (pred == val[i].label) correct++;
    }
    return 100.0f * (float)correct / (float)val.size();
}
