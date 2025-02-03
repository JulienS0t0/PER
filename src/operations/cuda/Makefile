NVCC = nvcc
NVFLAGS = -O2

SRC = $(wildcard $(CURDIR)/*.cu)
BIN_DIR = $(CURDIR)/../../../out/operations/cuda
BIN = $(patsubst $(CURDIR)/%.cu, $(BIN_DIR)/%, $(SRC))

.PHONY: all clean

all: $(BIN)

$(BIN_DIR)/%: $(CURDIR)/%.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVFLAGS) -o $@ $^

clean:
	rm -rf $(BIN_DIR)