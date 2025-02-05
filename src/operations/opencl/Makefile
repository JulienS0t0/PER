CXX = g++
CXXFLAGS = -Wall
LDFLAGS = -I/usr/include -L/usr/lib -lOpenCL # Potentiels flags de liaison

SRC = $(wildcard $(CURDIR)/*.cpp)
BIN_DIR = $(CURDIR)/../../../out/operations/opencl
RES_DIR = $(CURDIR)/../../../res/opencl
BIN = $(patsubst $(CURDIR)/%.cpp, $(BIN_DIR)/%, $(SRC))

MATRIX_UTILS_OBJ = $(CURDIR)/../../../out/matrices/matrix_utils.o

.PHONY: all clean clean-res

all: $(BIN)

$(BIN_DIR)/%: $(CURDIR)/%.cpp $(MATRIX_UTILS_OBJ)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(RES_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(RES_DIR)

clean-res:
	rm -rf $(RES_DIR)/*