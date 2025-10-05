# Makefile for Bone Fracture Detection CUDA Project
# Compile all 4 modules together

# Compiler
NVCC = nvcc

# Target executable name
TARGET = bone_fracture_cuda

# All source files (.cu files)
SOURCES = kernels.cu data_loader.cu model.cu main.cu

# OpenCV flags
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

# Compilation flags
NVCC_FLAGS = -std=c++17 -O2

# Default target - compiles everything
all: $(TARGET)

# Compile all .cu files into one executable
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET) $(OPENCV_FLAGS)
	@echo "Compilation successful! Executable: $(TARGET)"

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean compiled files
clean:
	rm -f $(TARGET)
	@echo "Cleaned up executable"

# Help message
help:
	@echo "Bone Fracture Detection CUDA Project"
	@echo "====================================="
	@echo "Available commands:"
	@echo "  make          - Compile the project"
	@echo "  make run      - Compile and run the program"
	@echo "  make clean    - Remove compiled executable"
	@echo "  make help     - Show this help message"

.PHONY: all run clean help