CXX = g++
CXXFLAGS = -Wall

SRC = $(wildcard $(CURDIR)/*.cpp)
BIN_DIR = $(CURDIR)/../../../out/operations/opencl
BIN = $(patsubst $(CURDIR)/%.cpp, $(BIN_DIR)/%, $(SRC))

.PHONY: all clean

all: $(BIN)

$(BIN_DIR)/%: $(CURDIR)/%.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lOpenCL

clean:
	rm -rf $(BIN_DIR)