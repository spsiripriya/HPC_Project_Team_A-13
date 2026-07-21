# Bone Fracture Detection Using CNN in C++

## Overview

This project implements a Convolutional Neural Network (CNN) from scratch in C++ for classifying bone X-ray images as either **Normal** or **Fractured**. The project uses OpenCV for image preprocessing and follows a modular architecture, implementing forward propagation, backpropagation, and model training without relying on deep learning frameworks such as TensorFlow or PyTorch.

---

## Features

- CNN implemented from scratch in C++
- Custom forward and backpropagation
- Bone fracture classification
- Image preprocessing using OpenCV
- Modular and reusable code structure
- Training and inference pipeline

---

### Project Status
Currently, a standard CNN has been implemented in C++.  

## Team Members and Responsibilities

| Member   | Role                          | Files |
|----------|-------------------------------|-------|
| Member 1 | Python Implementation         | Makefile, fracture-ai1.ipynb |
| Member 2 | Neural Network Layers         | layers.h, layers.cpp |
| Member 3 | Model Architecture & Training | utils.h, utils.cpp, model.h, model.cpp, training.h, training.cpp |
| Member 4 | Data Pipeline & Integration   | dataset.h, dataset.cpp, main.cpp |

## Future Work
- Distributed Training using OpenMPI to handle larger datasets efficiently  
- GPU Acceleration using CUDA for faster convolution and backpropagation  

## Build and Run
Compile and run the project using:
```bash
Compile using:
g++ main.cpp model.cpp layers.cpp dataset.cpp utils.cpp training.cpp \
    -o cnn_run $(pkg-config --cflags --libs opencv4) -std=c++17 -O2

Run using:
./cnn_run
