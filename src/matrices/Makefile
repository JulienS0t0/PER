CC = gcc
CFLAGS = -O2 -Wall

SRC = $(wildcard $(CURDIR)/*.c)
BIN_DIR = $(CURDIR)/../../out/matrices
BIN = $(patsubst $(CURDIR)/%.c, $(BIN_DIR)/%, $(SRC))

MATRIX_DIRS = $(BIN_DIR)/float $(BIN_DIR)/int

GEN_MATRICES_EXEC = $(BIN_DIR)/genmatrice

.PHONY: all clean gen

all: $(BIN)

$(BIN_DIR)/%: $(CURDIR)/%.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^

gen: all $(MATRIX_DIRS)
	@echo "Generating matrices..."
	@mkdir -p $(MATRIX_DIRS)
	$(GEN_MATRICES_EXEC)
	

clean:
	rm -rf $(BIN_DIR)