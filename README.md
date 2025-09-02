# Distributed and CUDA-Accelerated Deep Learning for Image Classification

## Project Overview
This project implements a complete Convolutional Neural Network (CNN) training system in C++ using OpenCV for image processing. The system classifies bone X-ray images as either *normal* or *fractured*.

### Key Features
- CNN implementation from scratch (no external ML frameworks)
- Custom backpropagation algorithm
- OpenCV integration for image preprocessing and augmentation
- Modular architecture with clear separation of concerns
- Real-time training progress monitoring

### Project Status
Currently, a standard CNN has been implemented in C++.

## Team Members and Responsibilities

| Member | Role | Files |
|--------|------|-------|
<<<<<<< HEAD
| Member 1 | Python Implementation | - |
=======
| Member 1 | Python Implementation | Makefile, fracture-ai1.ipynb |
>>>>>>> 8ab56ef (Updated README with project details)
| Member 2 | Neural Network Layers | layers.h, layers.cpp |
| Member 3 | Model Architecture & Training | utils.h, utils.cpp, model.h, model.cpp, training.h, training.cpp |
| Member 4 | Data Pipeline & Integration | dataset.h, dataset.cpp, main.cpp |

## Future Work
- *Distributed Training* using OpenMPI to handle larger datasets efficiently  
- *GPU Acceleration* using CUDA for faster convolution and backpropagation

## Build and Run
Compile and run the project using:
```bash
g++ main.cpp model.cpp layers.cpp dataset.cpp utils.cpp training.cpp -o cnn_run pkg-config --cflags --libs opencv4 -std=c++17 -O2

Run the project using:
<<<<<<< HEAD
./cnn_run
=======
./cnn_run
>>>>>>> 8ab56ef (Updated README with project details)
