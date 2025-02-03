CC = g++
CFLAGS = -O2 -Wall

SRC = $(wildcard $(CURDIR)/*.cpp)
BIN_DIR = $(CURDIR)/../../../out/operations/cpu
BIN = $(patsubst $(CURDIR)/%.cpp, $(BIN_DIR)/%, $(SRC))

.PHONY: all clean

all: $(BIN)

$(BIN_DIR)/%: $(CURDIR)/%.cpp
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -rf $(BIN_DIR)