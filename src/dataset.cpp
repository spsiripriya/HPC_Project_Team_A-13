

#include "dataset.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

std::vector<Sample> load_folder_samples(const std::string &folder, int label) {
    std::vector<Sample> out;
    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        std::cerr << "Warning: folder not found -> " << folder << std::endl;
        return out;
    }

    int count = 0;
    for (const auto &entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();

        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Skipping (cannot read): " << path << std::endl;
            continue;
        }
        cv::resize(img, img, cv::Size(64, 64)); // IMG_SIZE = 64
        img.convertTo(img, CV_32F, 1.0/255.0);

        std::vector<float> v(64 * 64);
        for (int r = 0; r < 64; ++r) {
            for (int c = 0; c < 64; ++c) {
                v[r * 64 + c] = img.at<float>(r, c);
            }
        }

        out.push_back({v, label});
        ++count;
    }
    std::cout << "Loaded " << count << " images from " << folder << "\n";
    return out;
}

std::vector<Sample> build_dataset_samples(const std::string &base_folder) {
    std::vector<Sample> ds;
    auto a = load_folder_samples(base_folder + "/normal", 0);
    auto b = load_folder_samples(base_folder + "/fractured", 1);
    ds.insert(ds.end(), a.begin(), a.end());
    ds.insert(ds.end(), b.begin(), b.end());
    std::cout << "Total samples from " << base_folder << " = " << ds.size() << "\n";
    return ds;
}

void shuffle_dataset(std::vector<Sample> &ds) {
    std::shuffle(ds.begin(), ds.end(), rng);
}
