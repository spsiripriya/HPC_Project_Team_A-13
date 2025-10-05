// data_loader.cu - Data loading and preprocessing
// Contributor: Person 2
// This module handles image loading, preprocessing, and dataset management

#include "data_loader.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

// Load images from a single folder with specified label
std::vector<Sample> load_folder(const std::string &folder, int label, int max_samples) {
    std::vector<Sample> out;
    if (!fs::exists(folder)) {
        std::cerr << "Warning: Folder does not exist: " << folder << std::endl;
        return out;
    }
    
    int count = 0;
    for (const auto &entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();
        
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Warning: Could not read image: " << path << std::endl;
            continue;
        }
        
        // Resize to target size
        cv::resize(img, img, cv::Size(IMG_SIZE, IMG_SIZE));
        img.convertTo(img, CV_32F, 1.0 / 255.0);
        
        // Normalize to mean=0, std=1
        cv::Scalar mean, stddev;
        cv::meanStdDev(img, mean, stddev);
        img = (img - mean[0]) / (stddev[0] + 1e-7);
        
        // Convert to flat vector
        std::vector<float> v(IMG_SIZE * IMG_SIZE);
        for (int r = 0; r < IMG_SIZE; ++r)
            for (int c = 0; c < IMG_SIZE; ++c)
                v[r * IMG_SIZE + c] = img.at<float>(r, c);
        
        out.push_back({v, label});
        if (max_samples > 0 && ++count >= max_samples) break;
    }
    return out;
}

// Build complete dataset from base directory
std::vector<Sample> build_dataset(const std::string &base, int max_per_class) {
    std::vector<Sample> ds;
    
    auto normal_samples = load_folder(base + "/normal", 0, max_per_class);
    auto fractured_samples = load_folder(base + "/fractured", 1, max_per_class);
    
    ds.insert(ds.end(), normal_samples.begin(), normal_samples.end());
    ds.insert(ds.end(), fractured_samples.begin(), fractured_samples.end());
    
    std::cout << "Loaded " << ds.size() << " samples (" 
              << normal_samples.size() << " normal, " 
              << fractured_samples.size() << " fractured)\n";
    
    return ds;
}

// Softmax function for probability computation
std::vector<float> softmax(const std::vector<float>& z) {
    float m = *std::max_element(z.begin(), z.end());
    std::vector<float> ex(z.size());
    float sum = 0.0f;
    
    for (size_t i = 0; i < z.size(); ++i) {
        ex[i] = std::exp(z[i] - m);
        sum += ex[i];
    }
    
    for (size_t i = 0; i < z.size(); ++i) {
        ex[i] /= (sum + 1e-12f);
    }
    
    return ex;
}