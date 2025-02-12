CC = gcc
CXX = g++
CFLAGS = -O2 -Wall
CXXFLAGS = -O2 -Wall
LDFLAGS =  # Potentiels flags de liaison

SRC_C = $(filter-out $(CURDIR)/matrix_utils.c, $(wildcard $(CURDIR)/*.c))
SRC_CPP = $(filter-out $(CURDIR)/matrix_utils.c, $(wildcard $(CURDIR)/*.cpp))
BIN_DIR = $(CURDIR)/../../out/matrices
BIN_C = $(patsubst $(CURDIR)/%.c, $(BIN_DIR)/%, $(SRC_C))
BIN_CPP = $(patsubst $(CURDIR)/%.cpp, $(BIN_DIR)/%, $(SRC_CPP))

MATRIX_UTILS_OBJ_GCC = $(BIN_DIR)/c_matrix_utils.o
MATRIX_UTILS_OBJ_GPP = $(BIN_DIR)/matrix_utils.o
MATRIX_DIRS = $(BIN_DIR)/float $(BIN_DIR)/int
GEN_MATRICES_EXEC = $(BIN_DIR)/genmatrice

.PHONY: all clean gen

all: $(MATRIX_UTILS_OBJ_GCC) $(MATRIX_UTILS_OBJ_GPP) $(BIN_C) $(BIN_CPP)
	@mkdir -p $(MATRIX_DIRS)

# Compilation de matrix_utils.c avec gcc
$(MATRIX_UTILS_OBJ_GCC): $(CURDIR)/matrix_utils.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Compilation de matrix_utils.c avec g++
$(MATRIX_UTILS_OBJ_GPP): $(CURDIR)/matrix_utils.c
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compilation des fichiers C avec gcc
$(BIN_DIR)/%: $(CURDIR)/%.c $(MATRIX_UTILS_OBJ_GCC)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compilation des fichiers C++ avec g++
$(BIN_DIR)/%: $(CURDIR)/%.cpp $(MATRIX_UTILS_OBJ_GPP)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gen: all $(MATRIX_DIRS)
	@echo "Generating matrices..."
	@mkdir -p $(MATRIX_DIRS)
	$(GEN_MATRICES_EXEC)

clean:
	rm -rf $(BIN_DIR)

# Here so that it does not complain 
clean-res:
