
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include "kernels.h"
#include "data_loader.h"
#include "model.h"

using Clock = std::chrono::high_resolution_clock;

// Configuration
const int EPOCHS = 20;
const float INITIAL_LR = 0.001f;

// Update these paths to match your dataset location
const std::string DATA_ROOT = "/media/hasini/CC0A416A0A41531E/B_TECH_SUBJ/Sem5/High Performance Cloud Computing/Project/bone_fract/dataset";
const std::string TRAIN_DIR = DATA_ROOT + "/train";
const std::string VAL_DIR = DATA_ROOT + "/val";
const std::string TEST_DIR = DATA_ROOT + "/test";

// Metrics structure for evaluation
struct Metrics {
    int tp = 0, tn = 0, fp = 0, fn = 0;
    
    void add(int pred, int actual) {
        if (pred == 1 && actual == 1) tp++;
        else if (pred == 0 && actual == 0) tn++;
        else if (pred == 1 && actual == 0) fp++;
        else fn++;
    }
    
    float accuracy() { 
        return 100.0f * (tp + tn) / (tp + tn + fp + fn + 1e-9f); 
    }
    
    float precision() { 
        return 100.0f * tp / (tp + fp + 1e-9f); 
    }
    
    float recall() { 
        return 100.0f * tp / (tp + fn + 1e-9f); 
    }
    
    float f1() {
        float p = precision(), r = recall();
        return 2 * p * r / (p + r + 1e-9f);
    }
    
    void print(const std::string& prefix) {
        std::cout << prefix << " Acc: " << accuracy() << "% "
                  << "Prec: " << precision() << "% "
                  << "Rec: " << recall() << "% "
                  << "F1: " << f1() << "%\n";
    }
};

// Evaluate model on dataset
Metrics evaluate(SimpleMLP &model, const std::vector<Sample>& ds) {
    Metrics m;
    for (const auto &s : ds) {
        auto logits = model.forward(s.img, false); // No dropout during eval
        auto probs = softmax(logits);
        int pred = (probs[1] > probs[0]) ? 1 : 0;
        m.add(pred, s.label);
    }
    return m;
}

int main() {
    std::cout << "=== Bone Fracture Detection with CUDA ===" << std::endl;
    std::cout << "Loading datasets...\n\n";
    
    // Load all datasets
    auto train_ds = build_dataset(TRAIN_DIR);
    auto val_ds = build_dataset(VAL_DIR);
    auto test_ds = build_dataset(TEST_DIR);
    
    if (train_ds.empty()) {
        std::cerr << "Error: No training data found!\n";
        std::cerr << "Please check the path: " << TRAIN_DIR << std::endl;
        return 1;
    }
    
    std::cout << "\n";
    
    // Initialize model
    SimpleMLP model(IMG_SIZE * IMG_SIZE, 64, 2);
    std::mt19937 rng(std::random_device{}());
    
    float best_val_f1 = 0.0f;
    
    // Training loop
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        std::shuffle(train_ds.begin(), train_ds.end(), rng);
        
        auto t0 = Clock::now();
        double loss_sum = 0.0;
        Metrics train_m;
        
        // Learning rate decay
        float lr = INITIAL_LR * std::pow(0.95f, epoch - 1);
        
        // Training on each sample
        for (size_t i = 0; i < train_ds.size(); ++i) {
            const auto &s = train_ds[i];
            
            // Forward pass
            auto logits = model.forward(s.img, true);
            auto probs = softmax(logits);
            
            // Compute loss
            float loss = -std::log(std::max(probs[s.label], 1e-9f));
            loss_sum += loss;
            
            // Track metrics
            int pred = (probs[1] > probs[0]) ? 1 : 0;
            train_m.add(pred, s.label);
            
            // Compute gradients
            std::vector<float> grad(2);
            grad[0] = probs[0];
            grad[1] = probs[1];
            grad[s.label] -= 1.0f;
            
            // Backward pass and update
            model.backward(grad);
            model.update(lr);
            
            // Progress update
            if ((i + 1) % 500 == 0) {
                std::cout << "Epoch " << epoch << " [" << (i+1) << "/" << train_ds.size() 
                          << "] loss: " << (loss_sum / (i+1))
                          << " acc: " << train_m.accuracy() << "%\n";
            }
        }
        
        auto t1 = Clock::now();
        float dur = std::chrono::duration<double>(t1 - t0).count();
        
        // Epoch summary
        std::cout << "\n=== Epoch " << epoch << " (lr=" << lr << ") ===\n";
        std::cout << "Training - Loss: " << (loss_sum / train_ds.size()) << " ";
        train_m.print("");
        
        // Validation
        if (!val_ds.empty()) {
            auto val_m = evaluate(model, val_ds);
            val_m.print("Validation -");
            
            if (val_m.f1() > best_val_f1) {
                best_val_f1 = val_m.f1();
                std::cout << "*** New best F1! ***\n";
            }
        }
        
        std::cout << "Time: " << dur << "s\n\n";
    }
    
    std::cout << "\n=== Training Complete ===\n";
    std::cout << "Best Validation F1: " << best_val_f1 << "%\n\n";
    
    // Final test set evaluation
    if (!test_ds.empty()) {
        std::cout << "=== FINAL TEST SET EVALUATION ===\n";
        auto test_m = evaluate(model, test_ds);
        std::cout << "Test Set Results:\n";
        test_m.print("  ");
        std::cout << "\nConfusion Matrix:\n";
        std::cout << "  True Positives (Fractured correctly identified): " << test_m.tp << "\n";
        std::cout << "  True Negatives (Normal correctly identified): " << test_m.tn << "\n";
        std::cout << "  False Positives (Normal misclassified as Fractured): " << test_m.fp << "\n";
        std::cout << "  False Negatives (Fractured misclassified as Normal): " << test_m.fn << "\n";
        std::cout << "\nTotal Test Samples: " << (test_m.tp + test_m.tn + test_m.fp + test_m.fn) << "\n";
    } else {
        std::cout << "\nNo test set found at: " << TEST_DIR << "\n";
        std::cout << "Using validation set as final evaluation:\n";
        auto final_val_m = evaluate(model, val_ds);
        final_val_m.print("Final ");
    }
    
    return 0;
}
