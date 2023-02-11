CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

INCS = $(wildcard $(INCLUDE_DIR)/*.h $(INCLUDE_DIR)/*.hpp)
INC_FLAGS = -I$(INCLUDE_DIR)

EXEC = $(BUILD_DIR)/main

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(INCS)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
