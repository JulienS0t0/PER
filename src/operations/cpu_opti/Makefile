CC = gcc
CFLAGS1 = -O2 -Wall
CFLAGS2 = -O3 -Wall
LDFLAGS =  # Potentiels flags de liaison

SRC = $(wildcard $(CURDIR)/*.c)
BIN_DIR1 = $(CURDIR)/../../../out/operations/cpu_opti_O2
BIN_DIR2 = $(CURDIR)/../../../out/operations/cpu_opti_O3
RES_DIR = $(CURDIR)/../../../res/cpu_opti
BIN1 = $(patsubst $(CURDIR)/%.c, $(BIN_DIR1)/%, $(SRC))
BIN2 = $(patsubst $(CURDIR)/%.c, $(BIN_DIR2)/%, $(SRC))

MATRIX_UTILS_OBJ = $(CURDIR)/../../../out/matrices/c_matrix_utils.o

.PHONY: all clean clean-res

all: $(BIN1) $(BIN2)

# Compilation des autres fichiers en exécutables, en liant matrix_utils.o
$(BIN_DIR1)/%: $(CURDIR)/%.c $(MATRIX_UTILS_OBJ)
	@mkdir -p $(BIN_DIR1)
	$(CC) $(CFLAGS1) -o $@ $^ $(LDFLAGS)

$(BIN_DIR2)/%: $(CURDIR)/%.c $(MATRIX_UTILS_OBJ)
	@mkdir -p $(BIN_DIR2)
	$(CC) $(CFLAGS2) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR1)
	rm -rf $(BIN_DIR2)
	rm -rf $(RES_DIR)

clean-res:
	rm -rf $(RES_DIR)/*