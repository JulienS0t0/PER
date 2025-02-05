CC = g++
CFLAGS = -O2 -Wall
LDFLAGS =  # Potentiels flags de liaison

SRC = $(filter-out $(CURDIR)/matrix_utils.c, $(wildcard $(CURDIR)/*.c))
BIN_DIR = $(CURDIR)/../../out/matrices
BIN = $(patsubst $(CURDIR)/%.c, $(BIN_DIR)/%, $(SRC))

MATRIX_UTILS_OBJ = $(BIN_DIR)/matrix_utils.o
MATRIX_DIRS = $(BIN_DIR)/float $(BIN_DIR)/int
GEN_MATRICES_EXEC = $(BIN_DIR)/genmatrice

.PHONY: all clean gen

all: $(BIN)
	@mkdir -p $(MATRIX_DIRS)

# Compilation de matrix_utils.c en un fichier objet
$(MATRIX_UTILS_OBJ): $(CURDIR)/matrix_utils.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Compilation des autres fichiers en exécutables, en liant matrix_utils.o
$(BIN_DIR)/%: $(CURDIR)/%.c $(MATRIX_UTILS_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gen: all $(MATRIX_DIRS)
	@echo "Generating matrices..."
	@mkdir -p $(MATRIX_DIRS)
	$(GEN_MATRICES_EXEC)

clean:
	rm -rf $(BIN_DIR)

# Here so that it does not complain 
clean-res:
	