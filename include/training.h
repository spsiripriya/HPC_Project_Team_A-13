#ifndef TRAINING_H
#define TRAINING_H

#include "model.h"
#include "dataset.h"

float evaluate(SimpleCNN &model, const std::vector<Sample>& val);

#endif
