CC = g++
CFLAGS = -O2 -Wall
LDFLAGS =  # Potentiels flags de liaison

SRC = $(wildcard $(CURDIR)/*.cpp)
BIN_DIR = $(CURDIR)/../../../out/operations/cpu
RES_DIR = $(CURDIR)/../../../res/cpu
BIN = $(patsubst $(CURDIR)/%.cpp, $(BIN_DIR)/%, $(SRC))

MATRIX_UTILS_OBJ = $(CURDIR)/../../../out/matrices/matrix_utils.o

.PHONY: all clean clean-res

all: $(BIN)

# Compilation des autres fichiers en exécutables, en liant matrix_utils.o
$(BIN_DIR)/%: $(CURDIR)/%.cpp $(MATRIX_UTILS_OBJ)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(RES_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(RES_DIR)

clean-res:
	rm -rf $(RES_DIR)/*