// data_loader.h - Header file for data loading
// Contributor: Person 2

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

const int IMG_SIZE = 64;

struct Sample {
    std::vector<float> img;
    int label;
};

// Function declarations
std::vector<Sample> load_folder(const std::string &folder, int label, int max_samples = -1);
std::vector<Sample> build_dataset(const std::string &base, int max_per_class = -1);
std::vector<float> softmax(const std::vector<float>& z);

#endif // DATA_LOADER_H