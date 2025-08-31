CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

SRC_DIR = src
BUILD_DIR = build
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TARGET = cnn_scratch_train

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(OPENCV_FLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV_FLAGS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
