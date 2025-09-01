
#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>

struct Sample {
    std::vector<float> img;
    int label;
};

std::vector<Sample> load_folder_samples(const std::string &folder, int label);
std::vector<Sample> build_dataset_samples(const std::string &base_folder);
void shuffle_dataset(std::vector<Sample> &ds);

#endif


