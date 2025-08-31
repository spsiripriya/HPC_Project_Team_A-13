
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>

#include "utils.h"
#include "layers.h"
#include "model.h"
#include "dataset.h"
#include "training.h"

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// Hyperparameters
const int IMG_SIZE = 64;
const int EPOCHS = 2;
const float LR = 0.001f;

// Dataset paths
const std::string DATA_ROOT = "/media/siri/New Volume/Engineering/5 th sem/Project Sem 5/HPC/Dataset/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification";
const std::string TRAIN_DIR = DATA_ROOT + "/train";
const std::string VAL_DIR = DATA_ROOT + "/val";

int main() {
    std::cout << "Loading training dataset from: " << TRAIN_DIR << std::endl;
    auto train_ds = build_dataset_samples(TRAIN_DIR);
    if (train_ds.empty()) {
        std::cerr << "No training data found. Exiting.\n";
        return 1;
    }

    std::vector<Sample> val_ds;
    if (fs::exists(VAL_DIR)) {
        std::cout << "Loading validation dataset from: " << VAL_DIR << std::endl;
        val_ds = build_dataset_samples(VAL_DIR);
    }

    shuffle_dataset(train_ds);

    SimpleCNN model;

    int N = (int)train_ds.size();
    std::cout << "Total training samples: " << N << std::endl;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        auto t0 = Clock::now();
        double epoch_loss = 0.0;
        int correct = 0;

        for (int i = 0; i < N; ++i) {
            const Sample &s = train_ds[i];

            auto logits = model.forward(s.img);
            auto probs = softmax_vec(logits);
            float loss = cross_entropy_loss(probs, s.label);
            epoch_loss += loss;

            int pred = std::max_element(probs.begin(), probs.end()) - probs.begin();
            if (pred == s.label) correct++;

            model.backward(logits, s.label);
            model.step(LR);

            if ((i + 1) % 200 == 0) {
                std::cout << "Epoch " << epoch << " progress: " << (i + 1) << "/" << N 
                         << " avg_loss: " << (epoch_loss / (i + 1)) << "\n";
            }
        }

        auto t1 = Clock::now();
        double dur = std::chrono::duration<double>(t1 - t0).count();
        float acc = 100.0f * (float)correct / (float)N;
        std::cout << "Epoch " << epoch << " complete. Avg Loss: " << (epoch_loss / N)
                  << " Train Acc: " << acc << "%  Time(s): " << dur << "\n";

        if (!val_ds.empty()) {
            float val_acc = evaluate(model, val_ds);
            std::cout << "  Val Acc: " << val_acc << "%\n";
        }
    }

    model.save("cnn_scratch_weights.txt");
    std::cout << "Saved model weights to cnn_scratch_weights.txt\n";
    return 0;
}
