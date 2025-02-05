NVCC = nvcc
NVFLAGS = -O2
LDFLAGS =  # Potentiels flags de liaison

SRC = $(wildcard $(CURDIR)/*.cu)
BIN_DIR = $(CURDIR)/../../../out/operations/cuda
RES_DIR = $(CURDIR)/../../../res/cuda
BIN = $(patsubst $(CURDIR)/%.cu, $(BIN_DIR)/%, $(SRC))

MATRIX_UTILS_OBJ = $(CURDIR)/../../../out/matrices/matrix_utils.o

.PHONY: all clean clean-res

all: $(BIN)


$(BIN_DIR)/%: $(CURDIR)/%.cu $(MATRIX_UTILS_OBJ)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(RES_DIR)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(RES_DIR)

clean-res:
	rm -rf $(RES_DIR)/*