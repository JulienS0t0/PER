# Chemins vers les autres Makefiles
SUBDIRS = src/matrices src/operations/cpu src/operations/cuda src/operations/opencl

# Emplacement des binaires
BINARIES = out/matrices out/operations/cpu out/operations/cuda out/operations/opencl

# Matrices
MATRICES = out/matrices/float/*.csv out/matrices/int/*.csv

# Commandes principales
.PHONY: all build gen run clean

all: build # gen

build:
	@for dir in $(SUBDIRS); do $(MAKE) -C $$dir; done
	
gen:
	@$(MAKE) -C src/matrices gen

run: 
	@echo "Usage: make run <target>"
	@echo "Example: make run add | make run all"

# TODO : implémenter un moyen de lancer tous les benchmarks 
# run-all: run-add

# run-add:
#     @echo "Running add for CPU..."
#     @./out/operations/cpu/add > res/cpu/add.csv
#     @echo "Running add for CUDA..."
#     @./out/operations/cuda/add > res/cuda/add.csv
#     @echo "Running add for OpenCL..."
#     @./out/operations/opencl/add > res/opencl/add.csv

clean:
	@for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done
	rm -rf out/operations/* out/matrices/*